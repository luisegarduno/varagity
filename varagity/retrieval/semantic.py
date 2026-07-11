"""Dense (semantic) retrieval over pgvector (spec §10.1 step 3, §11.2).

Embeds the query in e5 **query mode** — the asymmetric counterpart of the
passage-mode embedding used at ingest time — and runs the cosine top-k
search of :class:`~varagity.stores.vector_store.ContextualVectorDB`.
"""

from varagity.config import get_settings
from varagity.debug.show import check_verbose, v_retrieve
from varagity.models.embeddings import EmbeddingsClient
from varagity.models.registry import get_model
from varagity.retrieval.base import register
from varagity.stores.records import RetrievedChunk
from varagity.stores.vector_store import ContextualVectorDB


@register("semantic")
class SemanticRetriever:
    """Retriever backed by the pgvector cosine index.

    The registry instantiates it without arguments (no I/O at import time);
    dependencies then resolve from settings per call — a fresh store
    connection per query keeps a long-lived chat session robust. Tests and
    the eval harness inject their own store/client instead.
    """

    def __init__(
        self,
        *,
        store: ContextualVectorDB | None = None,
        embeddings: EmbeddingsClient | None = None,
    ) -> None:
        """Create the retriever.

        Args:
            store: Vector store to search; opened from settings per call
                when omitted.
            embeddings: Embeddings client for query encoding; resolved via
                the model registry when omitted.
        """
        self._store = store
        self._embeddings = embeddings

    def retrieve(self, query: str, k: int, verbose: int | None = None) -> list[RetrievedChunk]:
        """Retrieve the top-k chunks by cosine similarity.

        Args:
            query: The user's query; embedded in e5 query mode.
            k: Number of chunks to return.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The top-k chunks, best first, with cosine-similarity scores.

        Raises:
            ValueError: If ``verbose`` is invalid.
            openai.APIError: If query embedding still fails after retries.
            psycopg.OperationalError: If the vector store is unreachable.
        """
        verbose = check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
        embeddings = self._embeddings if self._embeddings is not None else get_model("embedding")
        query_vector = embeddings.embed_query(query, verbose=verbose)
        if self._store is not None:
            chunks = self._store.search(query_vector, k, verbose=verbose)
        else:
            with ContextualVectorDB() as store:
                chunks = store.search(query_vector, k, verbose=verbose)
        v_retrieve(chunks, verbose)
        return chunks
