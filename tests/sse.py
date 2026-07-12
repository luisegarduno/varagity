"""Shared SSE test helpers: parse a text/event-stream body into events."""

import json
from typing import Any


def parse_sse(body: str) -> list[tuple[str | None, Any]]:
    """Parse an SSE body into ``(event_name, decoded_data)`` tuples.

    Args:
        body: The complete ``text/event-stream`` response text.

    Returns:
        One tuple per event, in stream order; ``data`` is JSON-decoded when
        possible, the raw string otherwise. Comment-only frames (keepalives)
        are skipped.
    """
    events: list[tuple[str | None, Any]] = []
    for block in body.split("\n\n"):
        name: str | None = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
        if name is None and not data_lines:
            continue  # empty block or comment-only keepalive
        raw = "\n".join(data_lines)
        try:
            data = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            data = raw
        events.append((name, data))
    return events
