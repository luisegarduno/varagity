"""End-to-end pipeline tests: ingest → retrieve (3 methods) → answer.

Real pgvector Postgres **and** Elasticsearch via testcontainers (spec §15.1
"e2e" row: real stores); deterministic fake embeddings (hashed bag-of-words,
so lexically similar texts land near each other) and a scripted fake LLM
stand in for the GPU services. PDF parsing runs for real (Docling + OCR on
CPU — the corpus includes the Phase 7 fixture PDFs), so the first run
downloads Docling/EasyOCR models.

Select with ``pytest -m e2e`` (needs Docker).
"""

import hashlib
import math
import re
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import psycopg
import pytest
from elasticsearch import Elasticsearch
from testcontainers.elasticsearch import ElasticSearchContainer
from testcontainers.postgres import PostgresContainer

from varagity.generation.answer import answer_query, format_context
from varagity.ingest.loader import ingest_corpus
from varagity.retrieval.bm25 import BM25Retriever
from varagity.retrieval.hybrid import HybridRetriever
from varagity.retrieval.semantic import SemanticRetriever
from varagity.stores.bm25_store import ElasticsearchBM25
from varagity.stores.vector_store import ContextualVectorDB

pytestmark = pytest.mark.e2e

SCHEMA_PATH = Path(__file__).parents[2] / "varagity" / "stores" / "schema.sql"
CORPUS_PATH = Path(__file__).parents[1] / "fixtures" / "corpus"
DIM = 1024  # must match the schema's vector(1024)
ES_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:9.2.0"  # same as compose
BM25_INDEX = "varagity_e2e_bm25"

QUESTION = "How long is the kelp corridor between the Bruma and Cinza arrays?"
# The fact planted in tests/fixtures/corpus/tidal_grid.txt, quoted verbatim.
PLANTED_FACT = "1.8-kilometer"
SCRIPTED_ANSWER = (
    "The kelp corridor is a 1.8-kilometer strip of cultivated kelp. [SOURCE]: tidal_grid.txt"
)


def _bag_of_words_vector(text: str) -> list[float]:
    """Deterministic 1024-dim embedding: hashed token counts, L2-normalized.

    Unlike random per-text vectors, shared tokens produce shared components,
    so cosine similarity is meaningful and the planted chunk is retrievable.
    """
    vector = [0.0] * DIM
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        bucket = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % DIM
        vector[bucket] += 1.0
    norm = math.sqrt(sum(component * component for component in vector)) or 1.0
    return [component / norm for component in vector]


class FakeEmbeddings:
    """Deterministic stand-in for EmbeddingsClient (both e5 modes)."""

    def __init__(self) -> None:
        self.query_calls: list[str] = []

    def embed_passages(self, texts: list[str], verbose: int | None = None) -> list[list[float]]:
        return [_bag_of_words_vector(text) for text in texts]

    def embed_query(self, query: str, verbose: int | None = None) -> list[float]:
        self.query_calls.append(query)
        return _bag_of_words_vector(query)


class ScriptedLLM:
    """Stand-in for LLMClient: records prompts, returns a scripted response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
    ) -> str:
        assert [message["role"] for message in messages] == ["user"]
        self.prompts.append(messages[0]["content"])
        return self.response


@pytest.fixture(scope="session")
def pg_conninfo() -> Iterator[str]:
    """A pgvector Postgres with schema.sql applied, for the whole session."""
    with PostgresContainer("pgvector/pgvector:pg16") as container:
        conninfo = psycopg.conninfo.make_conninfo(
            host=container.get_container_host_ip(),
            port=int(container.get_exposed_port(5432)),
            dbname=container.dbname,
            user=container.username,
            password=container.password,
        )
        with psycopg.connect(conninfo, autocommit=True) as conn:
            conn.execute(SCHEMA_PATH.read_text())
        yield conninfo


@pytest.fixture(scope="session")
def es_url() -> Iterator[str]:
    """A single-node Elasticsearch for the whole session.

    Disk-watermark allocation checks are disabled: on a host whose disk is
    >90% full, ES's default percentage watermarks refuse to allocate the
    throwaway index's primary shard and every operation times out.
    """
    container = (
        ElasticSearchContainer(ES_IMAGE, mem_limit="2g")
        .with_env("discovery.type", "single-node")
        .with_env("ES_JAVA_OPTS", "-Xms512m -Xmx512m")
        .with_env("cluster.routing.allocation.disk.threshold_enabled", "false")
    )
    with container as es:
        yield f"http://{es.get_container_host_ip()}:{es.get_exposed_port(9200)}"


@pytest.fixture
def bm25_store(es_url: str) -> Iterator[ElasticsearchBM25]:
    """A BM25 store on a fresh index (deleted per test; ingest recreates it)."""
    with Elasticsearch(es_url) as raw:
        raw.indices.delete(index=BM25_INDEX, ignore_unavailable=True)
    with ElasticsearchBM25(url=es_url, index_name=BM25_INDEX) as store:
        yield store


@pytest.fixture
def pinned_settings(settings_env: Callable[..., None]) -> None:
    """Pin every pipeline setting so the machine's .env cannot leak in.

    ``CONTEXTUALIZE`` is off here: the walking-skeleton tests exercise the
    vanilla-RAG baseline (plan decision #2); the contextual variant flips it.
    """
    settings_env(
        ALLOWED_EXTENSIONS=".pdf,.txt,.md",
        CHUNKING_STRATEGY="recursive_character",
        CHUNK_SIZE=400,
        CHUNK_OVERLAP=50,
        CONTEXTUALIZE="false",
        EMBEDDING_MODEL="fake-bow-1024",
        RETRIEVAL_METHOD="hybrid",
        TOP_K=10,
        SEMANTIC_WEIGHT="0.8",
        BM25_WEIGHT="0.2",
    )


def test_walking_skeleton_ingest_to_grounded_answer(
    pinned_settings: None, pg_conninfo: str, bm25_store: ElasticsearchBM25, es_url: str
) -> None:
    """★ The Phase-4 milestone, automated: fixtures → both stores → Q&A state."""
    embeddings = FakeEmbeddings()
    llm = ScriptedLLM(f"<think>scanning the context…</think>{SCRIPTED_ANSWER}")

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    with ContextualVectorDB(pg_conninfo) as store:
        summary = ingest_corpus(
            str(CORPUS_PATH), store=store, bm25=bm25_store, embeddings=embeddings, verbose=0
        )
        assert summary.discovered == 7  # 3 text/md + 4 PDFs (Phase 7)
        assert summary.ingested == 6
        assert summary.no_text == 1  # blank_pages.pdf: nothing recoverable, 0-chunk row
        assert summary.failed == 0
        assert summary.chunks > 0

        # Dual-write parity: both stores hold exactly this run's chunks
        # (the live-stack success criterion, in-container).
        with Elasticsearch(es_url) as raw:
            assert int(raw.count(index=BM25_INDEX)["count"]) == summary.chunks

        retriever = SemanticRetriever(store=store, embeddings=embeddings)
        state = answer_query(QUESTION, retriever=retriever, llm=llm, verbose=0)  # type: ignore[arg-type]

    # The query was embedded exactly once, in query mode, with the raw text.
    assert embeddings.query_calls == [QUESTION]

    # The planted chunk was retrieved — and ranked first by the
    # deterministic bag-of-words similarity.
    assert any(PLANTED_FACT in chunk.content for chunk in state["retrieved"]), (
        "the planted kelp-corridor chunk was not retrieved"
    )
    assert "kelp corridor" in state["retrieved"][0].content

    # The answer prompt contains the planted evidence (spec §10.2 grounding).
    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "using ONLY the CONTEXT" in prompt
    assert PLANTED_FACT in prompt
    assert f"QUESTION: {QUESTION}" in prompt

    # The §10.1 state threads through, answer think-stripped.
    assert state["query"] == QUESTION
    assert state["retrieved"] and state["formatted_context"] == format_context(state["retrieved"])
    assert state["formatted_context"] in prompt
    assert state["answer"] == SCRIPTED_ANSWER


class ContextualizingLLM:
    """Fake LLM for ingest-time contextualization (Phase 5 e2e variant).

    Extracts the chunk from the spec §11.1 prompt and returns a blurb built
    from it, wrapped in a ``<think>`` stage to prove stripping end-to-end.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
    ) -> str:
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        chunk = prompt.split("<chunk>\n")[1].split("\n</chunk>")[0]
        return (
            "<think>situating…</think>From the coastal infrastructure notes, "
            f"regarding: {chunk[:40]}"
        )


def test_contextualized_ingest_stores_blurbs_and_still_answers(
    pinned_settings: None,
    settings_env: Callable[..., None],
    pg_conninfo: str,
    bm25_store: ElasticsearchBM25,
) -> None:
    """Phase-5/6 e2e variant: CONTEXTUALIZE on with a fake LLM.

    Asserts the plan's psql criteria in-container — every chunk's ``context``
    is non-null and ``contextualized_content`` starts with the blurb — and
    that the *same* contextualized text is what BM25 indexed (the spec §19.1
    "used for embedding **and** BM25" DoD row).
    """
    settings_env(CONTEXTUALIZE="true")
    embeddings = FakeEmbeddings()
    context_llm = ContextualizingLLM()

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    with ContextualVectorDB(pg_conninfo) as store:
        summary = ingest_corpus(
            str(CORPUS_PATH),
            store=store,
            bm25=bm25_store,
            embeddings=embeddings,
            llm=context_llm,  # type: ignore[arg-type]
            verbose=0,
        )
        assert summary.ingested == 6
        assert summary.chunks > 0
        assert len(context_llm.prompts) == summary.chunks  # one LLM call per chunk

        # The blurb-prefixed text is what the BM25 index searches.
        blurb_hits = bm25_store.search("coastal infrastructure notes", k=summary.chunks, verbose=0)
        assert len(blurb_hits) == summary.chunks  # every chunk carries the blurb
        for hit in blurb_hits:
            assert hit.contextualized_content.startswith("From the coastal infrastructure notes")

        with psycopg.connect(pg_conninfo) as conn:
            row = conn.execute("SELECT count(*), count(context) FROM chunks").fetchone()
            assert row is not None
            total, with_context = row
            assert total == summary.chunks
            assert with_context == total  # context IS NOT NULL for every chunk
            chunk_rows = conn.execute(
                "SELECT context, contextualized_content, content FROM chunks"
            ).fetchall()
        for context, contextualized, content in chunk_rows:
            assert context.startswith("From the coastal infrastructure notes")
            assert "<think>" not in context
            assert contextualized == f"{context}\n\n{content}"  # spec §9.4 composition

        # The Q&A path still grounds and answers over contextualized chunks.
        retriever = SemanticRetriever(store=store, embeddings=embeddings)
        answer_llm = ScriptedLLM(SCRIPTED_ANSWER)
        state = answer_query(QUESTION, retriever=retriever, llm=answer_llm, verbose=0)  # type: ignore[arg-type]

    assert any(PLANTED_FACT in chunk.content for chunk in state["retrieved"])
    assert all(chunk.context for chunk in state["retrieved"])  # blurbs round-trip
    # The retrieved blurbs reach the grounding prompt via [CONTEXT] blocks.
    assert "From the coastal infrastructure notes" in state["formatted_context"]
    assert state["answer"] == SCRIPTED_ANSWER


def test_second_run_is_idempotent_and_still_answers(
    pinned_settings: None, pg_conninfo: str, bm25_store: ElasticsearchBM25, es_url: str
) -> None:
    """Re-running the startup sequence skips unchanged files; Q&A still works."""
    embeddings = FakeEmbeddings()

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    with ContextualVectorDB(pg_conninfo) as store:
        first = ingest_corpus(
            str(CORPUS_PATH), store=store, bm25=bm25_store, embeddings=embeddings, verbose=0
        )
        second = ingest_corpus(
            str(CORPUS_PATH), store=store, bm25=bm25_store, embeddings=embeddings, verbose=0
        )
        assert first.ingested == 6
        assert second.ingested == 0
        assert second.skipped == 6
        assert second.no_text == 1  # the known-empty blank PDF re-warns, unparsed
        with Elasticsearch(es_url) as raw:  # skipped files were not re-indexed
            assert int(raw.count(index=BM25_INDEX)["count"]) == first.chunks

        retriever = SemanticRetriever(store=store, embeddings=embeddings)
        llm = ScriptedLLM("Lantern. [SOURCE]: aurora_station.md")
        state = answer_query(
            "What reactor powers the Aurora station?",
            retriever=retriever,
            llm=llm,  # type: ignore[arg-type]
            verbose=0,
        )
    assert any("Lantern" in chunk.content for chunk in state["retrieved"])
    assert state["answer"].startswith("Lantern")


def test_rare_keyword_retrieved_via_bm25_and_hybrid(
    pinned_settings: None, pg_conninfo: str, bm25_store: ElasticsearchBM25
) -> None:
    """★ The Phase-6 milestone: an exact rare term reaches the top via BM25.

    "Pelican-9" appears in exactly one fixture chunk; the keyword arm must
    surface it through both the ``bm25`` and ``hybrid`` retrievers, with the
    full metadata record hydrated from pgvector (citable source).
    """
    embeddings = FakeEmbeddings()

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    with ContextualVectorDB(pg_conninfo) as store:
        ingest_corpus(
            str(CORPUS_PATH), store=store, bm25=bm25_store, embeddings=embeddings, verbose=0
        )

        query = "Pelican-9 cargo capacity"
        bm25_chunks = BM25Retriever(bm25=bm25_store, store=store).retrieve(query, k=5, verbose=0)
        assert bm25_chunks, "bm25 retrieved nothing"
        assert "Pelican-9" in bm25_chunks[0].content  # exact-term match ranks first
        assert bm25_chunks[0].score > 0
        assert bm25_chunks[0].metadata["file_name"] == "aurora_station.md"  # hydrated

        hybrid_chunks = HybridRetriever(
            store=store, bm25=bm25_store, embeddings=embeddings
        ).retrieve(query, k=5, verbose=0)
        assert any("Pelican-9" in chunk.content for chunk in hybrid_chunks)
        # Fused scores are reciprocal-rank sums, bounded by the weight sum.
        assert all(0 < chunk.score <= 1.0 for chunk in hybrid_chunks)


def test_pdf_facts_are_retrievable_and_answerable(
    pinned_settings: None, pg_conninfo: str, bm25_store: ElasticsearchBM25
) -> None:
    """★ The Phase-7 milestone: PDF-only facts reach grounded answers.

    A digital-PDF-only fact and a scanned-PDF-only fact both flow through
    ingest → hybrid retrieval → grounded answer, with ``extraction``
    provenance intact on the retrieved metadata.
    """
    embeddings = FakeEmbeddings()

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    with ContextualVectorDB(pg_conninfo) as store:
        ingest_corpus(
            str(CORPUS_PATH), store=store, bm25=bm25_store, embeddings=embeddings, verbose=0
        )
        retriever = HybridRetriever(store=store, bm25=bm25_store, embeddings=embeddings)

        # Digital PDF (fast path): the Saltmere ceilometer fact.
        digital_answer = "Firefly measures up to 12,400 meters. [SOURCE]: saltmere_observatory.pdf"
        llm = ScriptedLLM(digital_answer)
        state = answer_query(
            "How high can the Firefly ceilometer measure cloud-base height?",
            retriever=retriever,
            llm=llm,  # type: ignore[arg-type]
            verbose=0,
        )
        digital_hits = [c for c in state["retrieved"] if "12,400 meters" in c.content]
        assert digital_hits, "the digital-PDF ceilometer chunk was not retrieved"
        assert digital_hits[0].metadata["file_type"] == "pdf"
        assert digital_hits[0].metadata["extraction"] == "text"  # no OCR involved
        assert digital_hits[0].metadata["page"] is not None
        assert "12,400 meters" in llm.prompts[0]  # grounded, not parroted
        assert state["answer"] == digital_answer

        # Scanned PDF (OCR fallback): the Moorhen dredging fact.
        scanned_answer = "Nine meters. [SOURCE]: moorhen_dredging_memo.pdf"
        llm = ScriptedLLM(scanned_answer)
        state = answer_query(
            "What depth did the dredger Moorhen clear the harbor channel to?",
            retriever=retriever,
            llm=llm,  # type: ignore[arg-type]
            verbose=0,
        )
        scanned_hits = [c for c in state["retrieved"] if "NINE METERS" in c.content.upper()]
        assert scanned_hits, "the scanned-PDF dredging chunk was not retrieved"
        assert scanned_hits[0].metadata["file_type"] == "pdf"
        assert scanned_hits[0].metadata["extraction"] == "ocr_fallback"  # OCR provenance
        assert "MOORHEN" in scanned_hits[0].content.upper()
        assert "NINE METERS" in llm.prompts[0].upper()
        assert state["answer"] == scanned_answer


def test_all_three_retrieval_methods_answer(
    pinned_settings: None, pg_conninfo: str, bm25_store: ElasticsearchBM25
) -> None:
    """Every RETRIEVAL_METHOD value drives the full Q&A pipeline cleanly."""
    embeddings = FakeEmbeddings()

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    with ContextualVectorDB(pg_conninfo) as store:
        ingest_corpus(
            str(CORPUS_PATH), store=store, bm25=bm25_store, embeddings=embeddings, verbose=0
        )
        retrievers = {
            "semantic": SemanticRetriever(store=store, embeddings=embeddings),
            "bm25": BM25Retriever(bm25=bm25_store, store=store),
            "hybrid": HybridRetriever(store=store, bm25=bm25_store, embeddings=embeddings),
        }
        for method, retriever in retrievers.items():
            llm = ScriptedLLM(SCRIPTED_ANSWER)
            state = answer_query(
                QUESTION,
                retriever=retriever,  # type: ignore[arg-type]
                llm=llm,  # type: ignore[arg-type]
                verbose=0,
            )
            assert state["retrieved"], f"{method} retrieved nothing"
            assert any(PLANTED_FACT in chunk.content for chunk in state["retrieved"]), (
                f"{method} missed the planted kelp-corridor chunk"
            )
            assert state["answer"] == SCRIPTED_ANSWER
            # provenance survives hydration for every method
            assert all(chunk.metadata.get("source") for chunk in state["retrieved"])
