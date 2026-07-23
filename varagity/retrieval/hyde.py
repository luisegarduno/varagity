"""HyDE — Hypothetical Document Embeddings over a base retriever (ADR-016).

The Gao et al. (2022) pattern adapted to the registry convention: one
non-streaming LLM call writes a *hypothetical answer passage* for the
query, and the **dense arm searches with that passage's embedding** instead
of the query's. The passage is embedded in e5 **passage mode**
(:meth:`~varagity.models.embeddings.EmbeddingsClient.embed_passages` — the
paper's document-encoder choice), landing it in the same vector space as
the ingested corpus, where passage↔passage neighbors are exactly the
chunks that *look like* the answer.

Like ``reranked``, HyDE *composes* a base retriever
(``HYDE_BASE_METHOD``: ``semantic`` | ``hybrid``) rather than forking
fusion — and the substitution is **dense-arm only**: the base receives the
user's original query text (a ``hybrid`` base's BM25 arm keeps its exact
keyword recall) plus the hypothetical passage's vector via the
``query_vector`` seam. Pairing with the cross-encoder stacks the other way
around: ``RETRIEVAL_METHOD=reranked`` + ``RERANK_BASE_METHOD=hyde``, so the
reranker judges candidates against the user's *real* query, never the
hypothetical (``hyde`` therefore refuses ``reranked`` as its own base —
config-validated).

Failure is a fallback, not an error (the condense-stage posture): a
transient LLM failure (after the client's own ``tenacity`` retries), an
empty cleaned passage, or an absurdly long one all degrade to the base
method's raw-query retrieval at ``WARNING``. ``HYDE_ENABLED=false`` is the
kill switch, checked inside the retriever exactly as ``RERANK_ENABLED`` is
— deliberately orthogonal to method selection. The generated passage
**must** pass through :func:`~varagity.models.llm.clean_response`: llama.cpp
emits ``<think>`` blocks and the non-streaming
:meth:`~varagity.models.llm.LLMClient.generate` does not strip them — an
unstripped one embedded as the search probe silently destroys retrieval.
"""

import logging
import time
from typing import cast

from varagity.config import get_settings
from varagity.debug.show import check_verbose, v_hyde, v_retrieve
from varagity.models.embeddings import EmbeddingsClient
from varagity.models.llm import LLMClient, clean_response
from varagity.models.registry import get_model
from varagity.observability import metrics
from varagity.retrieval.base import Retriever, get_retriever, register
from varagity.stores.records import RetrievedChunk

logger = logging.getLogger(__name__)

HYDE_PROMPT = """Write one short passage (3-5 sentences) that directly answers the question \
below, as it would appear in a reference document about the subject.
State the substance plainly — concrete facts, names, figures, and terms. If you do not know \
the true answer, write the most plausible passage anyway: the text is used only as a search \
probe against a document collection, never shown to anyone.
Reply with the passage only — no title, no quotes, no explanation.

QUESTION: {query}
PASSAGE:"""

# The completion-priming label the prompt ends with. Exposed so the
# retriever can strip a model's echo of it from the passage (the
# CONDENSE_QUERY_LABEL lesson — an echoed label would ride into the
# embedding model as noise); a test asserts the template still ends with
# it, so the two can't drift apart.
HYDE_PASSAGE_LABEL = "PASSAGE:"


@register("hyde")
class HydeRetriever:
    """LLM hypothetical passage → passage-mode embedding → base retriever.

    The registry instantiates it without arguments (no I/O at import time);
    the base retriever and model clients then resolve from settings per
    call, so a config change needs no restart. Tests and the eval harness
    inject their own instead.
    """

    def __init__(
        self,
        *,
        base: Retriever | None = None,
        llm: LLMClient | None = None,
        embeddings: EmbeddingsClient | None = None,
    ) -> None:
        """Create the retriever.

        Args:
            base: Retriever the hypothetical-passage vector is handed to;
                resolved from ``settings.HYDE_BASE_METHOD`` per call when
                omitted.
            llm: Chat client generating the passage; resolved via the model
                registry (``settings.HYDE_MODEL_TYPE``) per call when
                omitted.
            embeddings: Embeddings client for the passage-mode encoding;
                resolved via the model registry (``get_model("embedding")``)
                per call when omitted.
        """
        self._base = base
        self._llm = llm
        self._embeddings = embeddings

    def _base_retriever(self) -> Retriever:
        """Resolve the composed base retriever (injected or from settings).

        Returns:
            The base retriever instance.

        Raises:
            KeyError: If ``settings.HYDE_BASE_METHOD`` names an unregistered
                method (config validation makes this unreachable for
                env-sourced settings).
        """
        if self._base is not None:
            return self._base
        return get_retriever(get_settings().HYDE_BASE_METHOD)

    def _hypothetical(self, query: str, verbose: int) -> str | None:
        """Generate the hypothetical answer passage, or ``None`` to fall back.

        One non-streaming LLM call, post-processed exactly like the
        condense stage: ``<think>`` stages stripped
        (:func:`~varagity.models.llm.clean_response`), a prompt-label echo
        removed, then the empty/overlong guards. Every unusable outcome —
        including a transport failure after the client's own retries —
        returns ``None`` at ``WARNING`` so the caller degrades to the
        base method's raw-query retrieval instead of failing the turn.

        Args:
            query: The search query (the chat engine's ``search_query`` —
                already condensed when that stage ran).
            verbose: Validated console verbosity (0–2).

        Returns:
            The cleaned passage, or ``None`` when no usable passage was
            generated.
        """
        settings = get_settings()
        prompt = HYDE_PROMPT.format(query=query)
        # HYDE_MODEL_TYPE is validated to the LLM aliases, so the registry
        # always resolves an LLMClient here; the cast (rather than an
        # isinstance gate) keeps duck-typed test doubles usable at this
        # seam, matching the flows' injectable-fake convention.
        client = (
            self._llm
            if self._llm is not None
            else cast("LLMClient", get_model(settings.HYDE_MODEL_TYPE))
        )
        started = time.perf_counter()
        try:
            # verbose=0: the sub-call renders nothing; v_hyde below is this
            # stage's console output (the reranked-retriever pattern).
            raw = client.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=settings.HYDE_MAX_TOKENS,
                verbose=0,
            )
        except Exception:  # any failure falls back — the query must not die here
            logger.warning("HyDE LLM call failed — searching with the raw query", exc_info=True)
            return None
        passage = clean_response(raw)
        if passage.upper().startswith(HYDE_PASSAGE_LABEL):
            passage = passage[len(HYDE_PASSAGE_LABEL) :].strip()
        if not passage:
            logger.warning("HyDE returned an empty passage — searching with the raw query")
            return None
        if len(passage) > settings.HYDE_MAX_CHARS:
            logger.warning(
                "HyDE passage is %d chars (HYDE_MAX_CHARS=%d) — the generator "
                "misbehaved; searching with the raw query",
                len(passage),
                settings.HYDE_MAX_CHARS,
            )
            return None
        # The generation sub-stage's share of the embed stage ("is it
        # earning its latency?" — the rerank sub-stage pattern); the flow's
        # `embed` observation includes it.
        metrics.observe_query_stage("hyde", "hyde", time.perf_counter() - started)
        v_hyde(query, passage, verbose)
        return passage

    def encode_query(self, query: str, verbose: int | None = None) -> list[float] | None:
        """Encode the *hypothetical passage* the dense arm searches with.

        Generates the passage and embeds it in e5 **passage mode** — the
        vector its :meth:`retrieve` uses, per the protocol contract. With
        ``HYDE_ENABLED=false`` (logged), or whenever no usable passage was
        generated (the fallback), the base method's own query encoding is
        returned instead — retrieval then behaves exactly like the base.

        Args:
            query: The user's query, unformatted.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The passage-mode embedding of the hypothetical passage, or the
            base retriever's query encoding on the degraded paths.

        Raises:
            ValueError: If ``verbose`` is invalid.
            openai.APIError: If embedding still fails after retries.
        """
        settings = get_settings()
        verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
        if not settings.HYDE_ENABLED:
            logger.info(
                "HYDE_ENABLED=false — degrading to the %r base method's query encoding",
                settings.HYDE_BASE_METHOD,
            )
            return self._base_retriever().encode_query(query, verbose)
        passage = self._hypothetical(query, verbose)
        if passage is None:
            return self._base_retriever().encode_query(query, verbose)
        embeddings = self._embeddings if self._embeddings is not None else get_model("embedding")
        return embeddings.embed_passages([passage], verbose=0)[0]

    def retrieve(
        self,
        query: str,
        k: int,
        verbose: int | None = None,
        *,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k via the base method, dense arm steered by HyDE.

        The base retriever receives the user's **original query text** (a
        ``hybrid`` base's BM25 arm keeps exact keyword recall; a composing
        ``reranked`` cross-encodes the real query) together with the
        hypothetical passage's vector — the dense arm is the only arm HyDE
        redirects. Chunks and their traces pass through untouched: the
        base's ranking *is* the ranking.

        Args:
            query: The user's query; searched verbatim by every non-dense
                consumer.
            k: Number of chunks to return.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.
            query_vector: Pre-computed :meth:`encode_query` output (the
                flow's embed stage — already the hypothetical passage's
                vector); generated and encoded here when omitted.

        Returns:
            The base retriever's top-k chunks, best first.

        Raises:
            ValueError: If ``verbose`` is invalid.
            openai.APIError: If embedding still fails after retries.
        """
        verbose = check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
        if query_vector is None:
            query_vector = self.encode_query(query, verbose=verbose)
        chunks = self._base_retriever().retrieve(query, k=k, verbose=0, query_vector=query_vector)
        v_retrieve(chunks, verbose)
        return chunks
