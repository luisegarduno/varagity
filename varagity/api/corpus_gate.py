"""One writer per corpus, across every runner that writes it (ADR-019).

The ingest runner, the graph build runner, and the backup runner each keep
their own single-flight check, which stops a second run *of the same kind*
(``ingest_already_running``, ``graph_build_running``). A corpus needs more
than that once backups exist: an export must not snapshot a corpus a
restore is replacing, and a restore must not replace one an ingest is
writing. The gate is that shared rule — one :class:`CorpusGate` per corpus,
claimed for the whole of a run by whichever operation got there first, and
refused to everyone else with :class:`CorpusBusy` (the routes' ``409
corpus_busy``).

It is deliberately small and in-process: the API is the only process that
runs these operations through the GUI (the graph's single writer by design),
so a ``threading.Lock`` is the whole mechanism. Claims never block — a busy
corpus is an answer, not a wait.

**The cross-process half (RAG only).** Some RAG writers run in their own
process — ``uv run main.py ingest``, and ``main.py chat``, which ingests on
startup — where no in-process gate can see them. The RAG corpus therefore
also has a Postgres advisory lock
(:meth:`~varagity.stores.vector_store.ContextualVectorDB.corpus_lock`): every
ingest, in any process, holds it shared for its whole run, and a restore
holds it exclusive. The gate orders the API's own operations (and names the
holder in the ``409``); the lock is what keeps a CLI ingest and a restore
apart. The graph needs no such half — its single writer *is* this process.
"""

import threading
from collections.abc import Collection
from typing import Literal

Operation = Literal["ingest", "build", "export", "restore"]
"""Every operation that writes (or snapshots) a corpus."""

_ARTICLES: dict[str, str] = {"ingest": "an", "export": "an"}


class CorpusBusy(RuntimeError):
    """Raised by :meth:`CorpusGate.claim` while another operation holds the corpus.

    Attributes:
        corpus: The corpus that is busy.
        holder: The operation holding it — named in the message, so the
            ``409 corpus_busy`` says what to wait for.
    """

    def __init__(self, corpus: str, holder: Operation) -> None:
        """Name the busy corpus and its holder.

        Args:
            corpus: The corpus that is busy.
            holder: The operation holding it.
        """
        article = _ARTICLES.get(holder, "a")
        super().__init__(
            f"the {corpus} corpus is busy — {article} {holder} is running; "
            "try again when it finishes"
        )
        self.corpus = corpus
        self.holder: Operation = holder


class CorpusGate:
    """The non-blocking, single-holder write claim on one corpus.

    Attributes:
        corpus: The corpus this gate guards (``"rag"`` | ``"graph"``).
    """

    def __init__(self, corpus: str) -> None:
        """Create an unclaimed gate.

        Args:
            corpus: The corpus this gate guards.
        """
        self.corpus = corpus
        self._lock = threading.Lock()
        self._holder: Operation | None = None

    def claim(self, operation: Operation) -> None:
        """Take the corpus for ``operation``, or refuse at once.

        Args:
            operation: The operation starting.

        Raises:
            CorpusBusy: If any operation (the same kind included) already
                holds the corpus.
        """
        with self._lock:
            if self._holder is not None:
                raise CorpusBusy(self.corpus, self._holder)
            self._holder = operation

    def release(self) -> None:
        """Give the corpus back (idempotent: releasing a free gate is a no-op)."""
        with self._lock:
            self._holder = None

    def ensure_free(self) -> None:
        """Refuse at once if any operation holds the corpus — without claiming it.

        For routes that should answer ``409`` *before* doing expensive work
        (a restore hashing a bundle) or that write outside any run (the
        document deletes). The claim that follows can still lose a race;
        it raises the same error.

        Raises:
            CorpusBusy: If the corpus is held.
        """
        holder = self.holder
        if holder is not None:
            raise CorpusBusy(self.corpus, holder)

    def ensure_not_held_by(self, operations: Collection[Operation]) -> None:
        """Refuse at once if one of ``operations`` holds the corpus — without claiming it.

        :meth:`ensure_free` narrowed to the operations a caller actually
        conflicts with: a document delete is fine beside an ingest but not
        beside a backup, a graph source delete only clashes with a restore.

        Args:
            operations: The holders that refuse the caller.

        Raises:
            CorpusBusy: If one of them holds the corpus.
        """
        holder = self.holder
        if holder is not None and holder in operations:
            raise CorpusBusy(self.corpus, holder)

    @property
    def holder(self) -> Operation | None:
        """The operation holding the corpus right now.

        Returns:
            Its name, or ``None`` when the corpus is free.
        """
        with self._lock:
            return self._holder


_gates: dict[str, CorpusGate] = {"rag": CorpusGate("rag"), "graph": CorpusGate("graph")}


def get_corpus_gates() -> dict[str, CorpusGate]:
    """Provide the process-wide gates, keyed by corpus (FastAPI dependency, override seam).

    Returns:
        ``{"rag": …, "graph": …}`` — the same two objects for the life of
        the process, so every runner that claims a corpus shares its gate.
    """
    return _gates
