"""SSE event framing and the flow→stream bridge (spec_v2 §4.3).

Two concerns live here:

* **Typed framing** — one constructor per event in the chat protocol
  (``retrieval``, ``reasoning``, ``token``, ``done``, ``error``), each
  pairing the event name with its :mod:`varagity.api.schemas` payload, so
  routes can't emit a misnamed or mistyped frame.
* **The bridge** — the query pipeline is synchronous and runs in a worker
  thread (spec_v2 §4.1: async at the edge, sync flows underneath), while
  the SSE response is an async generator on the event loop.
  :class:`EventBridge` carries frames from the flow's callbacks to the
  generator (thread-safe, order-preserving) and carries the client's
  disconnect back to the flow as an abort flag polled between tokens.
"""

import asyncio
import logging
import threading
from collections.abc import AsyncIterator

from fastapi.sse import ServerSentEvent
from pydantic import BaseModel

from varagity.api.schemas import (
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    RetrievalEvent,
)

logger = logging.getLogger(__name__)

EVENT_RETRIEVAL = "retrieval"
EVENT_REASONING = "reasoning"
EVENT_TOKEN = "token"
EVENT_DONE = "done"
EVENT_ERROR = "error"


def retrieval_event(payload: RetrievalEvent) -> ServerSentEvent:
    """Frame the provenance payload as the ``retrieval`` event.

    Args:
        payload: The retrieved chunks and method metadata.

    Returns:
        The framed event.
    """
    return ServerSentEvent(event=EVENT_RETRIEVAL, data=payload)


def delta_event(kind: str, text: str) -> ServerSentEvent:
    """Frame one classified text fragment as ``reasoning`` or ``token``.

    Args:
        kind: The splitter's classification — ``"reasoning"`` maps to the
            ``reasoning`` event, ``"answer"`` to ``token``.
        text: The fragment.

    Returns:
        The framed event.
    """
    event = EVENT_REASONING if kind == "reasoning" else EVENT_TOKEN
    return ServerSentEvent(event=event, data=DeltaEvent(delta=text))


def done_event(payload: DoneEvent) -> ServerSentEvent:
    """Frame the terminal ``done`` event.

    Args:
        payload: Message/conversation ids, the full answer, and usage.

    Returns:
        The framed event.
    """
    return ServerSentEvent(event=EVENT_DONE, data=payload)


def error_event(code: str, message: str) -> ServerSentEvent:
    """Frame an in-band ``error`` event (mid-stream failure).

    Args:
        code: Stable, machine-readable code.
        message: Human-readable detail.

    Returns:
        The framed event.
    """
    return ServerSentEvent(event=EVENT_ERROR, data=ErrorEvent(code=code, message=message))


class EventBridge:
    """Order-preserving conduit between a worker-thread flow and an SSE stream.

    Built on the running event loop; the flow's callbacks call :meth:`emit`
    from their thread (marshalled with ``call_soon_threadsafe``), the async
    generator drains :meth:`events`, and the stream side signals the flow to
    stop with :meth:`abort` (polled via :meth:`should_abort` between
    tokens — how a closed tab frees the GPU, spec_v2 §4.3 cancellation).

    Must be constructed on the event loop thread.
    """

    def __init__(self) -> None:
        """Capture the running loop and start with an empty queue."""
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[ServerSentEvent | None] = asyncio.Queue()
        self._abort = threading.Event()

    def emit(self, event: str, payload: BaseModel) -> None:
        """Enqueue one frame from any thread.

        Args:
            event: The SSE event name.
            payload: The event's typed payload.
        """
        self._send(ServerSentEvent(event=event, data=payload))

    def emit_frame(self, frame: ServerSentEvent) -> None:
        """Enqueue an already-framed event from any thread.

        Args:
            frame: The framed event.
        """
        self._send(frame)

    def close(self) -> None:
        """Signal end-of-stream; :meth:`events` finishes after the backlog."""
        self._send(None)

    def abort(self) -> None:
        """Ask the producing flow to stop at its next between-tokens check."""
        self._abort.set()

    def should_abort(self) -> bool:
        """Whether :meth:`abort` was called (the flow's poll seam).

        Returns:
            ``True`` once the stream side requested a stop.
        """
        return self._abort.is_set()

    async def events(self) -> AsyncIterator[ServerSentEvent]:
        """Drain frames in emission order until :meth:`close`.

        Yields:
            Each enqueued frame.
        """
        while (frame := await self._queue.get()) is not None:
            yield frame

    def _send(self, item: ServerSentEvent | None) -> None:
        """Marshal one queue item onto the loop, tolerating shutdown races.

        Args:
            item: The frame, or ``None`` as the end-of-stream sentinel.
        """
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:  # loop already closed (shutdown) — nothing to deliver to
            logger.debug("dropped an SSE frame emitted after the event loop closed")
