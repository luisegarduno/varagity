"""Shared store lifecycle plumbing.

Every persistent store in this package owns a single client or connection and
exposes an idempotent :meth:`~ClosingContextMixin.close`.
:class:`ClosingContextMixin` supplies the identical ``with``-statement wiring
they all need — enter returns the store, exit closes it — so each store only
writes the ``close`` body that actually differs (a pgvector connection, an
Elasticsearch client).
"""

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self


class ClosingContextMixin(ABC):
    """Context-manager plumbing over a store's own :meth:`close`.

    Mixed into the pgvector, Elasticsearch, settings, and conversation
    stores: entering the ``with`` block yields the store, leaving it calls
    :meth:`close`. ``__exit__`` returns ``None``, so an exception raised
    inside the block propagates unchanged.
    """

    @abstractmethod
    def close(self) -> None:
        """Release the underlying client or connection (idempotent)."""

    def __enter__(self) -> Self:
        """Enter a context that closes the store on exit.

        Returns:
            This store.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the store on context exit.

        Args:
            exc_type: Exception type, if the block raised.
            exc: Exception instance, if the block raised.
            tb: Traceback, if the block raised.
        """
        self.close()
