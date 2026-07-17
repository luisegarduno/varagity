"""Typed application configuration.

Settings load from the process environment and the repo-root ``.env`` file
(see ``.env.example``). Modules read the :class:`Settings` object obtained via
:func:`get_settings` — never ``os.getenv`` — so configuration stays typed,
validated, and mockable in tests.
"""

from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the environment and ``.env``.

    ``.env`` also carries lowercase Docker Compose interpolation variables
    (e.g. ``embeddings_volume``), so unknown keys are ignored rather than
    rejected.

    Defaults are the in-container values (see ``.env.example``); host runs
    override them via ``.env`` per the host-vs-container comment convention.

    Attributes:
        LOG_LEVEL: Root logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
        DEFAULT_VERBOSE: Default console verbosity for public functions;
            ``0`` = off, ``1`` = low (names, counts), ``2`` = high (full
            metadata, panels).
        DOCS_PATH: Directory scanned for the ingest corpus.
        ALLOWED_EXTENSIONS: Comma-separated whitelist of ingestable file
            extensions (v2 widens the v1 ``.pdf``/``.txt``/``.md`` set with
            the office/web formats — spec_v2 §8.1).
        PDF_OCR_FALLBACK: Whether a PDF whose text-layer extraction comes
            up (near-)empty is automatically re-converted with OCR (plan
            decision #10). Off = pass 1's result stands and a textless PDF
            ends in the empty-extraction guard.
        PDF_OCR_MIN_CHARS: Below this many non-whitespace characters a
            pass-1 extraction counts as "no text layer" and triggers the
            OCR fallback.
        PDF_OCR_TEXTLESS_PAGE_RATIO: Textless-page share at or above which
            the OCR fallback triggers (catches mixed scanned/digital
            documents whose digital pages alone pass the length check).
        PDF_OCR_FORCE_FULL_PAGE: Escape hatch for corrupt-text-layer PDFs
            (garbage embedded text passes the content triggers): skip pass
            1 and OCR every page, ignoring embedded text. Off by default —
            digital pages of mixed documents keep their text layer.
        OCR_ENGINE: Registry name of the OCR engine used by the fallback
            (``easyocr`` | ``tesseract``; see
            ``varagity.ingest.parsers.pdf.OCR_ENGINE_FACTORIES``). EasyOCR
            is the benchmark-decided default (ADR-004): error-free on the
            fixture scans where Tesseract drops words; Tesseract trades
            that accuracy for ~5× throughput.
        OCR_LANGUAGES: Comma-separated ISO 639-1 language codes for OCR,
            primary language first (mapped per engine, e.g. ``en`` →
            Tesseract's ``eng``).
        PREVIEW_ENABLED: Kill switch for the evidence-panel page preview
            (ADR-010) — the ``RERANK_ENABLED`` pattern: off, the preview
            routes degrade (``available:false, reason:"preview_disabled"``)
            and the GUI falls back to the full-text view. Env-only by
            design (not runtime-overridable).
        PREVIEW_RENDER_WIDTH: Rendered page-image width in pixels for
            ``GET /api/documents/{doc_id}/preview/page/{page}`` (512–4096;
            height follows the page's aspect ratio).
        PREVIEW_MIN_COVERAGE: Word-trigram coverage floor in ``[0.0, 1.0]``
            below which a preview locate reports ``no_match`` instead of
            guessing a page.
        PREVIEW_CONVERT_TIMEOUT_S: Timeout in seconds for one LibreOffice
            PPTX→PDF conversion (the preview path's only subprocess).
        CHUNKING_STRATEGY: Registry name of the chunking strategy
            (``recursive_character`` | ``token_based`` | ``markdown_aware``
            | ``docling_hybrid`` | ``semantic`` — see ``varagity.chunking``;
            spec_v2 §7). Changing it doesn't change content hashes, so an
            unchanged corpus needs ``ingest --reingest`` to re-chunk.
        CHUNK_SIZE: Chunk size budget. The **unit is per strategy**
            (each module documents its own): characters for
            ``recursive_character`` and ``markdown_aware`` (spec §9.3);
            **tokens** for ``token_based``, ``docling_hybrid``, and
            ``semantic`` (aligned to e5's 512-token ceiling — spec_v2 §7).
        CHUNK_OVERLAP: Overlap between consecutive chunks, in the same
            unit as ``CHUNK_SIZE`` for that strategy (unused by
            ``docling_hybrid``, which merges instead of overlapping).
        CONTEXTUALIZE: Whether ingestion generates an LLM situating blurb per
            chunk (Contextual Retrieval, spec §9.4). ``False`` keeps the
            identity path (``contextualized_content = content``) — the
            non-contextual eval baseline and a throughput knob (plan
            decision #2). Toggling it does not change content hashes, so
            re-processing an unchanged corpus needs ``ingest --reingest``.
        EMBEDDING_MODEL: Served model name passed to the embeddings API (the
            infinity ``INFINITY_SERVED_MODEL_NAME`` string, verbatim).
        EMBEDDING_API_URL: OpenAI-compatible base URL of the infinity server.
        EMBEDDING_API_KEY: Bearer token for the infinity server.
        EMBEDDING_DIM: Embedding dimensionality (1024 for e5-large-instruct).
        EMBEDDING_BATCH_SIZE: Number of passages sent per embeddings request.
        BASE_MODEL: Filename of the llama.cpp ``.gguf`` model, relative to the
            bind-mounted ``${models_volume}`` directory.
        BASE_MODEL_API_URL: OpenAI-compatible base URL of the llama.cpp server.
        BASE_MODEL_API_KEY: Bearer token for the llama.cpp server (it runs
            unauthenticated in v1, but the OpenAI client requires a value).
        MAX_TOKENS: Generation cap per LLM response.
        LLM_TEMPERATURE: Sampling temperature for LLM responses.
        LLM_CONTEXT_TOKENS: The served model's context window, in tokens —
            must mirror the llama.cpp ``--ctx-size`` in
            ``docker-compose.yml``. llama.cpp (context shift disabled, its
            default) fails a request with a hard 500 once prompt + generated
            tokens reach the window, so generation caps are clamped against
            this value (:class:`varagity.models.llm.LLMClient`).
        CONTEXTUALIZE_MAX_TOKENS: Generation cap for the per-chunk situating
            blurbs (spec §9.4) — sized for a reasoning preamble plus a short
            blurb, deliberately far below the chat-sized ``MAX_TOKENS``: an
            8k reserve against a 16k window rejects any document over ~8k
            tokens. A blurb that overruns the cap degrades to "no context"
            via the empty-blurb path, never a failed ingest.
        CHAT_MODEL_TYPE: Model-registry type the chat surfaces (API + GUI)
            resolve their LLM with (``default`` | ``reasoning`` | ``tool``
            — the LLM aliases of :func:`varagity.models.registry.get_model`;
            spec_v2 §4.7's Generation "model type" knob). All three resolve
            to the single llama.cpp server in v2; the knob exists so the
            composer quick-toggle survives the aliases becoming separate
            servers post-v2.
        CHAT_ENGINE: Registry name of the chat engine preparing each turn's
            retrieval query (spec_v3 §4.2; see ``varagity.chat``). ``simple``
            searches with the user's words, verbatim (the stateless v2
            behavior); ``condense_context`` rewrites follow-ups into
            standalone search queries against the conversation history.
        CONDENSE_ENABLED: Kill switch for the condense stage, orthogonal to
            engine selection (spec_v3 §4.6 — the ``RERANK_ENABLED``
            pattern): with ``CHAT_ENGINE=condense_context`` and this off,
            the engine degrades to ``simple`` behavior (the identity split)
            and logs it.
        CONDENSE_MODEL_TYPE: Model-registry type the condense stage
            resolves its LLM with when the caller injects none (``default``
            | ``reasoning`` | ``tool`` — mirrors ``CHAT_MODEL_TYPE``). All
            three are one llama.cpp server today; the knob is the seam for
            pointing condensing at a smaller model without a code change.
        CONDENSE_HISTORY_TURNS: Maximum prior messages loaded for and fed
            to the condenser (6 = three user/assistant pairs). ``0``
            disables history — every turn then takes the no-history path
            and never condenses.
        CONDENSE_MAX_TOKENS: Generation cap for the condense call — a
            standalone search query is a sentence, not an essay.
        CONDENSE_MAX_CHARS: Length ceiling on the cleaned condense output;
            anything longer means the condenser misbehaved, and the raw
            query is searched instead (the spec_v3 §4.6 fallback).
        POSTGRES_HOST: PostgreSQL host (service name in-container).
        POSTGRES_PORT: PostgreSQL port.
        POSTGRES_DB: PostgreSQL database name.
        POSTGRES_USER: PostgreSQL user.
        POSTGRES_PASSWORD: PostgreSQL password (dev-only static credential).
        ELASTICSEARCH_URL: Base URL of the Elasticsearch server (BM25 store).
        BM25_INDEX_NAME: Name of the contextual BM25 index.
        PREFECT_API_URL: Base URL of the Prefect server's API. Exported to
            the process environment by ``varagity.pipeline`` **before**
            ``prefect`` is imported (Prefect captures its environment at
            import time), so flow/task runs are tracked by the compose
            ``prefect`` service.
        RETRIEVAL_METHOD: Registry name of the retrieval method (spec §10.1
            + spec_v2 §5: ``semantic`` | ``bm25`` | ``hybrid`` |
            ``reranked``; the default is ``hybrid``).
        TOP_K: Number of chunks retrieved per query.
        SEMANTIC_WEIGHT: Hybrid rank-fusion weight of the semantic (pgvector)
            arm (spec §11.4). Must sum to 1.0 with ``BM25_WEIGHT``.
        BM25_WEIGHT: Hybrid rank-fusion weight of the BM25 (Elasticsearch)
            arm. Must sum to 1.0 with ``SEMANTIC_WEIGHT``.
        RERANK_ENABLED: Kill switch for the cross-encoder stage, orthogonal
            to method selection (spec_v2 §5.2): with
            ``RETRIEVAL_METHOD=reranked`` and this off, the ``reranked``
            retriever degrades to its base method's ranking and logs it.
        RERANK_MODEL: Served reranker name passed to the infinity
            ``/rerank`` endpoint, verbatim. Must be a cross-encoder
            (``bge-reranker-v2-m3``); infinity structurally rejects
            bi-encoders (e5, jina) at ``/rerank``.
        RERANK_API_URL: OpenAI-style base URL of the infinity server
            serving the reranker (``/rerank`` lives under the same ``/v1``
            prefix as embeddings).
        RERANK_API_KEY: Bearer token for the reranker (the same infinity
            key as embeddings).
        RERANK_TOP_N: Documents kept after re-ranking. Must be positive and
            ≤ ``RERANK_CANDIDATES``; with ``RETRIEVAL_METHOD=reranked``
            also ≤ ``TOP_K`` (rerank narrows; it can't invent candidates).
        RERANK_BASE_METHOD: Registry name of the retriever the ``reranked``
            method composes (``semantic`` | ``bm25`` | ``hybrid``; not
            ``reranked`` — no recursion).
        RERANK_CANDIDATES: Candidate-pool size over-fetched from the base
            retriever and cross-encoded per query (v2 plan decision #3 —
            the Anthropic cookbook's 150→20 over-fetch, scaled to this
            corpus).
        API_HOST: Bind address of the HTTP API server (spec_v2 §4.1).
        API_PORT: Port of the HTTP API server.
        API_CORS_ORIGINS: Comma-separated origins allowed CORS access to
            the API (the web app's origin; dev-only posture — spec_v2 §14).
        UPLOAD_MAX_MB: Per-file size cap for corpus uploads
            (``POST /api/documents``, spec_v2 §4.2).
        UPLOAD_MAX_FILES: Per-request file-count cap for corpus uploads
            (spec_v3 §5.2) — a dragged home directory gets a clean 422
            instead of a filled disk.
        UPLOAD_MAX_TOTAL_MB: Per-request size cap summed across every file
            of one upload (``UPLOAD_MAX_MB`` stays the per-file cap; this
            bounds the batch).
        UPLOAD_MAX_PATH_DEPTH: Maximum path segments (folders plus the file
            name) a relative upload path may carry (spec_v3 §5.2 — folder
            uploads preserve structure, bounded).
        METRICS_ENABLED: Whether the API serves ``GET /metrics`` (spec_v2
            §6). Gates the endpoint only — the in-app collectors always
            record (cheap in-memory counters).
        PROMETHEUS_PORT: Host port the compose ``prometheus`` service
            binds (compose interpolation; the app never dials it).
        GRAFANA_PORT: Host port the compose ``grafana`` service binds
            (compose interpolation; the container serves 3000 internally).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LOG_LEVEL: str = "INFO"
    DEFAULT_VERBOSE: int = 1

    DOCS_PATH: str = "./docs"
    ALLOWED_EXTENSIONS: str = ".pdf,.txt,.md,.docx,.pptx,.xlsx,.html,.htm"

    PDF_OCR_FALLBACK: bool = True
    PDF_OCR_MIN_CHARS: int = 50
    PDF_OCR_TEXTLESS_PAGE_RATIO: float = 0.2
    PDF_OCR_FORCE_FULL_PAGE: bool = False
    OCR_ENGINE: str = "easyocr"  # benchmark-decided default (ADR-004)
    OCR_LANGUAGES: str = "en"

    PREVIEW_ENABLED: bool = True
    PREVIEW_RENDER_WIDTH: int = 1536
    PREVIEW_MIN_COVERAGE: float = 0.3
    PREVIEW_CONVERT_TIMEOUT_S: int = 120

    CHUNKING_STRATEGY: str = "recursive_character"
    CHUNK_SIZE: int = 400  # characters, not tokens (spec §9.3)
    CHUNK_OVERLAP: int = 50

    CONTEXTUALIZE: bool = True

    EMBEDDING_MODEL: str = "infloat/multilingual-e5-large-instruct"
    EMBEDDING_API_URL: str = "http://infinity-embeddings:8081/v1"
    EMBEDDING_API_KEY: str = "change-me"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32

    BASE_MODEL: str = "Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf"
    BASE_MODEL_API_URL: str = "http://llamacpp:8080/v1"
    BASE_MODEL_API_KEY: str = "none"
    MAX_TOKENS: int = 8192
    LLM_TEMPERATURE: float = 0.6
    LLM_CONTEXT_TOKENS: int = 16384
    CONTEXTUALIZE_MAX_TOKENS: int = 2048
    CHAT_MODEL_TYPE: str = "default"
    CHAT_ENGINE: str = "simple"
    CONDENSE_ENABLED: bool = True
    CONDENSE_MODEL_TYPE: str = "default"
    CONDENSE_HISTORY_TURNS: int = 6
    CONDENSE_MAX_TOKENS: int = 128
    CONDENSE_MAX_CHARS: int = 512

    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "varagity"
    POSTGRES_USER: str = "varagity"
    POSTGRES_PASSWORD: str = "change-me"

    ELASTICSEARCH_URL: str = "http://elasticsearch:9200"
    BM25_INDEX_NAME: str = "varagity_contextual_bm25"

    PREFECT_API_URL: str = "http://prefect:4200/api"

    RETRIEVAL_METHOD: str = "hybrid"
    TOP_K: int = 10
    SEMANTIC_WEIGHT: float = 0.8
    BM25_WEIGHT: float = 0.2

    RERANK_ENABLED: bool = False
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_API_URL: str = "http://infinity-embeddings:8081/v1"
    RERANK_API_KEY: str = "change-me"
    RERANK_TOP_N: int = 5
    RERANK_BASE_METHOD: str = "hybrid"
    RERANK_CANDIDATES: int = 40

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_CORS_ORIGINS: str = "http://localhost:3000"
    UPLOAD_MAX_MB: int = 50
    UPLOAD_MAX_FILES: int = 500
    UPLOAD_MAX_TOTAL_MB: int = 2048
    UPLOAD_MAX_PATH_DEPTH: int = 12

    METRICS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 9090
    GRAFANA_PORT: int = 3001

    @property
    def allowed_extension_set(self) -> frozenset[str]:
        """Parsed ``ALLOWED_EXTENSIONS`` as a normalized set.

        Entries are lowercased, stripped, and guaranteed to start with a dot
        (``"md"`` and ``".md"`` are equivalent in the env value).

        Returns:
            The allowed extensions, e.g. ``frozenset({".pdf", ".txt", ".md"})``.
        """
        extensions = set()
        for raw in self.ALLOWED_EXTENSIONS.split(","):
            ext = raw.strip().lower()
            if not ext:
                continue
            extensions.add(ext if ext.startswith(".") else f".{ext}")
        return frozenset(extensions)

    @property
    def ocr_language_list(self) -> list[str]:
        """Parsed ``OCR_LANGUAGES`` as an ordered, deduplicated list.

        Order is preserved (OCR engines weight the primary language);
        entries are lowercased and stripped.

        Returns:
            The language codes, e.g. ``["en", "de"]``.
        """
        languages: list[str] = []
        for raw in self.OCR_LANGUAGES.split(","):
            code = raw.strip().lower()
            if code and code not in languages:
                languages.append(code)
        return languages

    @property
    def cors_origin_list(self) -> list[str]:
        """Parsed ``API_CORS_ORIGINS`` as an ordered, deduplicated list.

        Entries are stripped; trailing slashes are removed (an origin is
        scheme://host[:port], never a path).

        Returns:
            The allowed origins, e.g. ``["http://localhost:3000"]``.
        """
        origins: list[str] = []
        for raw in self.API_CORS_ORIGINS.split(","):
            origin = raw.strip().rstrip("/")
            if origin and origin not in origins:
                origins.append(origin)
        return origins

    @model_validator(mode="after")
    def _validate_api(self) -> "Settings":
        """Reject HTTP API/observability parameters outside their domains (spec_v2 §10).

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If ``API_PORT``, ``PROMETHEUS_PORT``, or
                ``GRAFANA_PORT`` is not a valid TCP port, or
                ``UPLOAD_MAX_MB`` is not positive.
        """
        for name in ("API_PORT", "PROMETHEUS_PORT", "GRAFANA_PORT"):
            port = getattr(self, name)
            if not 0 < port < 65536:
                raise ValueError(f"{name} must be a valid TCP port (1–65535); got {port}")
        if self.UPLOAD_MAX_MB <= 0:
            raise ValueError(f"UPLOAD_MAX_MB must be positive; got {self.UPLOAD_MAX_MB}")
        return self

    @model_validator(mode="after")
    def _validate_upload_limits(self) -> "Settings":
        """Reject upload batch caps that cannot admit a valid batch (spec_v3 §5.2).

        ``UPLOAD_MAX_MB``'s own positivity lives in :meth:`_validate_api`;
        this validator owns the batch-level caps and their one cross
        constraint: a per-file cap above the whole batch's budget is a
        config bug, not a tighter limit.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If ``UPLOAD_MAX_FILES``, ``UPLOAD_MAX_TOTAL_MB``, or
                ``UPLOAD_MAX_PATH_DEPTH`` is not positive, or if
                ``UPLOAD_MAX_MB`` exceeds ``UPLOAD_MAX_TOTAL_MB``.
        """
        for name in ("UPLOAD_MAX_FILES", "UPLOAD_MAX_TOTAL_MB", "UPLOAD_MAX_PATH_DEPTH"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive; got {getattr(self, name)}")
        if self.UPLOAD_MAX_MB > self.UPLOAD_MAX_TOTAL_MB:
            raise ValueError(
                f"UPLOAD_MAX_MB ({self.UPLOAD_MAX_MB}) must not exceed "
                f"UPLOAD_MAX_TOTAL_MB ({self.UPLOAD_MAX_TOTAL_MB}) — a per-file cap "
                "above the whole batch's budget is a config bug"
            )
        return self

    @field_validator("OCR_LANGUAGES")
    @classmethod
    def _validate_ocr_languages(cls, value: str) -> str:
        """Reject an OCR language list with no usable entries.

        Args:
            value: The configured ``OCR_LANGUAGES`` value.

        Returns:
            The validated value, unchanged.

        Raises:
            ValueError: If no non-empty code remains after splitting.
        """
        if not any(part.strip() for part in value.split(",")):
            raise ValueError("OCR_LANGUAGES must list at least one language code, e.g. 'en'")
        return value

    @model_validator(mode="after")
    def _validate_pdf_ocr(self) -> "Settings":
        """Reject OCR-fallback trigger parameters outside their domains.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If ``PDF_OCR_MIN_CHARS`` is negative or
                ``PDF_OCR_TEXTLESS_PAGE_RATIO`` is not within ``[0.0, 1.0]``.
        """
        if self.PDF_OCR_MIN_CHARS < 0:
            raise ValueError(
                f"PDF_OCR_MIN_CHARS must be non-negative; got {self.PDF_OCR_MIN_CHARS}"
            )
        if not 0.0 <= self.PDF_OCR_TEXTLESS_PAGE_RATIO <= 1.0:
            raise ValueError(
                "PDF_OCR_TEXTLESS_PAGE_RATIO must be between 0.0 and 1.0; "
                f"got {self.PDF_OCR_TEXTLESS_PAGE_RATIO}"
            )
        return self

    @model_validator(mode="after")
    def _validate_preview(self) -> "Settings":
        """Reject page-preview parameters outside their domains (ADR-010).

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If ``PREVIEW_RENDER_WIDTH`` is outside 512–4096,
                ``PREVIEW_MIN_COVERAGE`` is outside ``[0.0, 1.0]``, or
                ``PREVIEW_CONVERT_TIMEOUT_S`` is not positive.
        """
        if not 512 <= self.PREVIEW_RENDER_WIDTH <= 4096:
            raise ValueError(
                "PREVIEW_RENDER_WIDTH must be between 512 and 4096 pixels; "
                f"got {self.PREVIEW_RENDER_WIDTH}"
            )
        if not 0.0 <= self.PREVIEW_MIN_COVERAGE <= 1.0:
            raise ValueError(
                f"PREVIEW_MIN_COVERAGE must be between 0.0 and 1.0; got {self.PREVIEW_MIN_COVERAGE}"
            )
        if self.PREVIEW_CONVERT_TIMEOUT_S <= 0:
            raise ValueError(
                f"PREVIEW_CONVERT_TIMEOUT_S must be positive; got {self.PREVIEW_CONVERT_TIMEOUT_S}"
            )
        return self

    @field_validator("ALLOWED_EXTENSIONS")
    @classmethod
    def _validate_allowed_extensions(cls, value: str) -> str:
        """Reject an extension whitelist with no usable entries.

        Args:
            value: The configured ``ALLOWED_EXTENSIONS`` value.

        Returns:
            The validated value, unchanged.

        Raises:
            ValueError: If no non-empty extension remains after splitting.
        """
        if not any(part.strip() for part in value.split(",")):
            raise ValueError("ALLOWED_EXTENSIONS must list at least one extension, e.g. '.txt,.md'")
        return value

    @model_validator(mode="after")
    def _validate_sizes(self) -> "Settings":
        """Reject size parameters that cannot produce a valid pipeline run.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If ``CHUNK_SIZE``, ``EMBEDDING_DIM``,
                ``EMBEDDING_BATCH_SIZE``, ``MAX_TOKENS``, ``TOP_K``,
                ``LLM_CONTEXT_TOKENS``, or ``CONTEXTUALIZE_MAX_TOKENS`` is
                not positive, if ``CHUNK_OVERLAP`` is negative, if
                ``CHUNK_OVERLAP`` is not smaller than ``CHUNK_SIZE``, or if
                ``MAX_TOKENS``/``CONTEXTUALIZE_MAX_TOKENS`` is not smaller
                than ``LLM_CONTEXT_TOKENS`` (a generation cap that fills the
                whole window leaves no room for any prompt).
        """
        positives = (
            "CHUNK_SIZE",
            "EMBEDDING_DIM",
            "EMBEDDING_BATCH_SIZE",
            "MAX_TOKENS",
            "TOP_K",
            "LLM_CONTEXT_TOKENS",
            "CONTEXTUALIZE_MAX_TOKENS",
        )
        for name in positives:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive; got {getattr(self, name)}")
        if self.CHUNK_OVERLAP < 0:
            raise ValueError(f"CHUNK_OVERLAP must be non-negative; got {self.CHUNK_OVERLAP}")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be smaller than "
                f"CHUNK_SIZE ({self.CHUNK_SIZE})"
            )
        for name in ("MAX_TOKENS", "CONTEXTUALIZE_MAX_TOKENS"):
            if getattr(self, name) >= self.LLM_CONTEXT_TOKENS:
                raise ValueError(
                    f"{name} ({getattr(self, name)}) must be smaller than "
                    f"LLM_CONTEXT_TOKENS ({self.LLM_CONTEXT_TOKENS}) — a generation "
                    "cap that fills the whole context window leaves no room for "
                    "any prompt"
                )
        return self

    @field_validator("RETRIEVAL_METHOD")
    @classmethod
    def _validate_retrieval_method(cls, value: str) -> str:
        """Reject retrieval methods outside the spec §10.1 / spec_v2 §5 vocabulary.

        Membership in the vocabulary is validated here; registry membership
        is enforced by :func:`varagity.retrieval.get_retriever` at lookup
        time.

        Args:
            value: The configured ``RETRIEVAL_METHOD`` value.

        Returns:
            The validated value, unchanged.

        Raises:
            ValueError: If ``value`` is not ``semantic``, ``bm25``,
                ``hybrid``, or ``reranked``.
        """
        allowed = ("semantic", "bm25", "hybrid", "reranked")
        if value not in allowed:
            raise ValueError(f"RETRIEVAL_METHOD must be one of {allowed}; got {value!r}")
        return value

    @model_validator(mode="after")
    def _validate_rerank(self) -> "Settings":
        """Reject rerank parameters that cannot produce a valid rerank stage.

        The cross-method constraints (spec_v2 §5.3): re-ranking *narrows* a
        wider candidate pool, so ``RERANK_TOP_N`` can never exceed the pool,
        and when the ``reranked`` method is selected it can't promise more
        results than ``TOP_K`` nor rerank a pool smaller than ``TOP_K``.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If ``RERANK_TOP_N`` is not in
                ``(0, RERANK_CANDIDATES]``, if ``RERANK_BASE_METHOD`` is not
                ``semantic``/``bm25``/``hybrid`` (``reranked`` would
                recurse), or if ``RETRIEVAL_METHOD == "reranked"`` with
                ``RERANK_TOP_N > TOP_K`` or ``RERANK_CANDIDATES < TOP_K``.
        """
        if not 0 < self.RERANK_TOP_N <= self.RERANK_CANDIDATES:
            raise ValueError(
                f"RERANK_TOP_N ({self.RERANK_TOP_N}) must be positive and at most "
                f"RERANK_CANDIDATES ({self.RERANK_CANDIDATES})"
            )
        allowed_bases = ("semantic", "bm25", "hybrid")
        if self.RERANK_BASE_METHOD not in allowed_bases:
            raise ValueError(
                f"RERANK_BASE_METHOD must be one of {allowed_bases} (not 'reranked' — "
                f"no recursion); got {self.RERANK_BASE_METHOD!r}"
            )
        if self.RETRIEVAL_METHOD == "reranked":
            if self.RERANK_TOP_N > self.TOP_K:
                raise ValueError(
                    f"RERANK_TOP_N ({self.RERANK_TOP_N}) must not exceed TOP_K "
                    f"({self.TOP_K}) when RETRIEVAL_METHOD is 'reranked' — rerank "
                    "narrows; it can't invent candidates"
                )
            if self.RERANK_CANDIDATES < self.TOP_K:
                raise ValueError(
                    f"RERANK_CANDIDATES ({self.RERANK_CANDIDATES}) must be at least "
                    f"TOP_K ({self.TOP_K}) when RETRIEVAL_METHOD is 'reranked'"
                )
        return self

    @model_validator(mode="after")
    def _validate_fusion_weights(self) -> "Settings":
        """Reject hybrid rank-fusion weights that don't form a convex blend.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If either weight is negative, or if
                ``SEMANTIC_WEIGHT + BM25_WEIGHT`` is not 1.0 (spec §6; checked
                with a small tolerance because the values arrive as decimal
                strings, e.g. ``0.7 + 0.3`` is not exactly ``1.0`` in binary
                floating point).
        """
        for name in ("SEMANTIC_WEIGHT", "BM25_WEIGHT"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative; got {getattr(self, name)}")
        total = self.SEMANTIC_WEIGHT + self.BM25_WEIGHT
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"SEMANTIC_WEIGHT ({self.SEMANTIC_WEIGHT}) + BM25_WEIGHT ({self.BM25_WEIGHT}) "
                f"must sum to 1.0; got {total}"
            )
        return self

    @field_validator("CHAT_MODEL_TYPE")
    @classmethod
    def _validate_chat_model_type(cls, value: str) -> str:
        """Reject chat model types outside the LLM aliases.

        The vocabulary is hard-coded (mirroring ``RETRIEVAL_METHOD``) because
        importing the model registry here would be circular — the registry's
        clients read this module. ``varagity.models.registry.LLM_MODEL_TYPES``
        is the same tuple, regression-tested to match.

        Args:
            value: The configured ``CHAT_MODEL_TYPE`` value.

        Returns:
            The validated value, unchanged.

        Raises:
            ValueError: If ``value`` is not ``default``, ``reasoning``, or
                ``tool`` (``embedding``/``rerank`` are not chat models).
        """
        allowed = ("default", "reasoning", "tool")
        if value not in allowed:
            raise ValueError(f"CHAT_MODEL_TYPE must be one of {allowed}; got {value!r}")
        return value

    @field_validator("CHAT_ENGINE")
    @classmethod
    def _validate_chat_engine(cls, value: str) -> str:
        """Reject chat engines outside the spec_v3 §4.2 vocabulary.

        The vocabulary is hard-coded (mirroring ``CHAT_MODEL_TYPE``) because
        importing the engine registry here would be circular —
        ``varagity/chat/`` reads this module through its model clients. The
        tuple is regression-tested to equal
        ``varagity.chat.CHAT_ENGINE_REGISTRY`` and grows in lockstep with
        it.

        Args:
            value: The configured ``CHAT_ENGINE`` value.

        Returns:
            The validated value, unchanged.

        Raises:
            ValueError: If ``value`` is not a registered engine name.
        """
        allowed = ("simple", "condense_context")
        if value not in allowed:
            raise ValueError(f"CHAT_ENGINE must be one of {allowed}; got {value!r}")
        return value

    @model_validator(mode="after")
    def _validate_condense(self) -> "Settings":
        """Reject condense-stage parameters outside their domains (spec_v3 §4.6).

        ``CONDENSE_MODEL_TYPE``'s vocabulary is hard-coded for the same
        circular-import reason as :meth:`_validate_chat_model_type` — the
        model registry's clients read this module.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If ``CONDENSE_HISTORY_TURNS`` is negative, if
                ``CONDENSE_MAX_TOKENS`` or ``CONDENSE_MAX_CHARS`` is not
                positive, or if ``CONDENSE_MODEL_TYPE`` is not ``default``,
                ``reasoning``, or ``tool``.
        """
        if self.CONDENSE_HISTORY_TURNS < 0:
            raise ValueError(
                f"CONDENSE_HISTORY_TURNS must be non-negative (0 disables history); "
                f"got {self.CONDENSE_HISTORY_TURNS}"
            )
        for name in ("CONDENSE_MAX_TOKENS", "CONDENSE_MAX_CHARS"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive; got {getattr(self, name)}")
        allowed = ("default", "reasoning", "tool")
        if self.CONDENSE_MODEL_TYPE not in allowed:
            raise ValueError(
                f"CONDENSE_MODEL_TYPE must be one of {allowed}; got {self.CONDENSE_MODEL_TYPE!r}"
            )
        return self

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def _validate_llm_temperature(cls, value: float) -> float:
        """Reject sampling temperatures outside the OpenAI-compatible range.

        Args:
            value: The configured ``LLM_TEMPERATURE`` value.

        Returns:
            The validated value, unchanged.

        Raises:
            ValueError: If ``value`` is not within ``[0.0, 2.0]``.
        """
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"LLM_TEMPERATURE must be between 0.0 and 2.0; got {value}")
        return value

    @field_validator("DEFAULT_VERBOSE")
    @classmethod
    def _validate_default_verbose(cls, value: int) -> int:
        """Reject verbosity defaults outside the supported levels.

        Args:
            value: The configured ``DEFAULT_VERBOSE`` value.

        Returns:
            The validated value, unchanged.

        Raises:
            ValueError: If ``value`` is not 0, 1, or 2.
        """
        if value not in (0, 1, 2):
            raise ValueError(f"DEFAULT_VERBOSE must be 0 (off), 1 (low), or 2 (high); got {value}")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` instance.

    The instance is created on first call and cached. Tests that need a
    fresh load (e.g. after patching the environment) call
    ``get_settings.cache_clear()`` or construct :class:`Settings` directly.

    Returns:
        The cached application settings.
    """
    return Settings()
