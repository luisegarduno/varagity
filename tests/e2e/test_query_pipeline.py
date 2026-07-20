"""End-to-end pipeline tests: ingest → retrieve (3 methods) → answer.

The pipelines run through the Prefect flows
(``varagity.pipeline``) under ``prefect_test_harness`` — every stage below
is a tracked task run against an ephemeral API, exactly the production
composition. Real pgvector Postgres **and** Elasticsearch via
testcontainers (spec §15.1 "e2e" row: real stores); deterministic fake
embeddings (hashed bag-of-words, so lexically similar texts land near each
other) and a scripted fake LLM stand in for the GPU services. PDF parsing
runs for real (Docling + OCR on CPU — the corpus includes the fixture
PDFs), so the first run downloads Docling/EasyOCR models.

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
from prefect.testing.utilities import prefect_test_harness

from varagity.chat import get_chat_engine
from varagity.chat.base import Turn
from varagity.eval.containers import ephemeral_elasticsearch, ephemeral_postgres
from varagity.generation.answer import format_context
from varagity.pipeline import ingest_flow, query_flow
from varagity.retrieval.bm25 import BM25Retriever
from varagity.retrieval.hybrid import HybridRetriever
from varagity.retrieval.semantic import SemanticRetriever
from varagity.stores.bm25_store import ElasticsearchBM25
from varagity.stores.vector_store import ContextualVectorDB

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module", autouse=True)
def prefect_harness() -> Iterator[None]:
    """Ephemeral Prefect API so the flows run tracked, hermetically."""
    with prefect_test_harness():
        yield


CORPUS_PATH = Path(__file__).parents[1] / "fixtures" / "corpus"
DIM = 1024  # must match the schema's vector(1024)
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
    with ephemeral_postgres() as conninfo:
        yield conninfo


@pytest.fixture(scope="session")
def es_url() -> Iterator[str]:
    """A single-node Elasticsearch for the whole session."""
    with ephemeral_elasticsearch() as url:
        yield url


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
    """★ The walking-skeleton milestone, automated: fixtures → both stores → Q&A state."""
    embeddings = FakeEmbeddings()
    llm = ScriptedLLM(f"<think>scanning the context…</think>{SCRIPTED_ANSWER}")

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    with ContextualVectorDB(pg_conninfo) as store:
        summary = ingest_flow(
            str(CORPUS_PATH), store=store, bm25=bm25_store, embeddings=embeddings, verbose=0
        )
        assert summary.discovered == 7  # 3 text/md + 4 PDFs
        assert summary.ingested == 6
        assert summary.no_text == 1  # blank_pages.pdf: nothing recoverable, 0-chunk row
        assert summary.failed == 0
        assert summary.chunks > 0

        # Dual-write parity: both stores hold exactly this run's chunks
        # (the live-stack success criterion, in-container).
        with Elasticsearch(es_url) as raw:
            assert int(raw.count(index=BM25_INDEX)["count"]) == summary.chunks

        retriever = SemanticRetriever(store=store, embeddings=embeddings)
        state = query_flow(QUESTION, retriever=retriever, llm=llm, verbose=0)  # type: ignore[arg-type]

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
    """Fake LLM for ingest-time contextualization.

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
    """Contextualized-ingest e2e variant: CONTEXTUALIZE on with a fake LLM.

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
        summary = ingest_flow(
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
        state = query_flow(QUESTION, retriever=retriever, llm=answer_llm, verbose=0)  # type: ignore[arg-type]

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
        first = ingest_flow(
            str(CORPUS_PATH), store=store, bm25=bm25_store, embeddings=embeddings, verbose=0
        )
        second = ingest_flow(
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
        state = query_flow(
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
    """★ An exact rare term reaches the top via BM25.

    "Pelican-9" appears in exactly one fixture chunk; the keyword arm must
    surface it through both the ``bm25`` and ``hybrid`` retrievers, with the
    full metadata record hydrated from pgvector (citable source).
    """
    embeddings = FakeEmbeddings()

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    with ContextualVectorDB(pg_conninfo) as store:
        ingest_flow(
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
    """★ PDF-only facts reach grounded answers.

    A digital-PDF-only fact and a scanned-PDF-only fact both flow through
    ingest → hybrid retrieval → grounded answer, with ``extraction``
    provenance intact on the retrieved metadata.
    """
    embeddings = FakeEmbeddings()

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    with ContextualVectorDB(pg_conninfo) as store:
        ingest_flow(
            str(CORPUS_PATH), store=store, bm25=bm25_store, embeddings=embeddings, verbose=0
        )
        retriever = HybridRetriever(store=store, bm25=bm25_store, embeddings=embeddings)

        # Digital PDF (fast path): the Saltmere ceilometer fact.
        digital_answer = "Firefly measures up to 12,400 meters. [SOURCE]: saltmere_observatory.pdf"
        llm = ScriptedLLM(digital_answer)
        state = query_flow(
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
        state = query_flow(
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


def test_office_web_facts_are_retrievable_and_answerable(
    pinned_settings: None,
    settings_env: Callable[..., None],
    pg_conninfo: str,
    bm25_store: ElasticsearchBM25,
) -> None:
    """★ Office/web-only facts reach grounded answers.

    With the widened whitelist, a fact answerable **only** from each new
    format (``.docx``/``.pptx``/``.xlsx``/``.html``) flows through ingest →
    hybrid retrieval → grounded answer, with format-true ``file_type``/
    ``page``/``extraction`` provenance on the retrieved metadata.
    """
    settings_env(ALLOWED_EXTENSIONS=".pdf,.txt,.md,.docx,.pptx,.xlsx,.html,.htm")
    embeddings = FakeEmbeddings()

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    with ContextualVectorDB(pg_conninfo) as store:
        summary = ingest_flow(
            str(CORPUS_PATH), store=store, bm25=bm25_store, embeddings=embeddings, verbose=0
        )
        assert summary.discovered == 11  # 3 text/md + 4 PDFs + 3 office + 1 web
        assert summary.ingested == 10
        assert summary.no_text == 1  # blank_pages.pdf, as ever
        assert summary.failed == 0

        retriever = HybridRetriever(store=store, bm25=bm25_store, embeddings=embeddings)
        cases = [
            (
                "At what distance before docking does the Gullwing ferry switch to battery power?",
                "800 meters",
                "gullwing_ferry_manual.docx",
                "docx",
                None,
            ),
            (
                "How much power does the Petrel-6 tidal turbine produce at peak flow?",
                "3.4 megawatts",
                "petrel_turbine_briefing.pptx",
                "pptx",
                1,  # slide → page (the plan's non-null .pptx criterion)
            ),
            (
                "How many mooring bollards are stored at the East Quay?",
                "Mooring bollard",
                "quayside_inventory.xlsx",
                "xlsx",
                1,  # sheet → page
            ),
            (
                "By how much did the seagrass meadow off Wrenhaven expand in 2025?",
                "12 hectares",
                "seagrass_survey.html",
                "html",
                None,
            ),
        ]
        for question, fact, file_name, file_type, page in cases:
            answer = f"{fact}. [SOURCE]: {file_name}"
            llm = ScriptedLLM(answer)
            state = query_flow(question, retriever=retriever, llm=llm, verbose=0)  # type: ignore[arg-type]
            hits = [c for c in state["retrieved"] if fact in c.content]
            assert hits, f"the {file_type} chunk with {fact!r} was not retrieved"
            assert hits[0].metadata["file_name"] == file_name
            assert hits[0].metadata["file_type"] == file_type
            assert hits[0].metadata["page"] == page
            assert hits[0].metadata["extraction"] == "text"  # digital text, never OCR
            assert fact in llm.prompts[0]  # grounded, not parroted
            assert state["answer"] == answer


def test_every_chunking_strategy_answers_a_planted_fact(
    pinned_settings: None,
    settings_env: Callable[..., None],
    pg_conninfo: str,
    bm25_store: ElasticsearchBM25,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ Every registered chunking strategy answers end-to-end.

    A corpus ingested under each new chunking strategy flows through ingest →
    hybrid retrieval → grounded answer on the planted kelp-corridor fact, with
    ``chunking_strategy`` provenance on every stored chunk and — for the
    heading-aware strategies — ``heading_path`` breadcrumbs in the metadata
    JSONB. Text-only corpus: per-strategy reingest must not re-pay PDF OCR.
    """
    settings_env(ALLOWED_EXTENSIONS=".txt,.md")
    embeddings = FakeEmbeddings()
    # The semantic chunker resolves its boundary-detection embeddings via the
    # model registry at split time; point it at the same deterministic fake.
    monkeypatch.setattr("varagity.chunking.semantic.get_model", lambda model_type: embeddings)

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    heading_aware = {"markdown_aware", "docling_hybrid"}
    with ContextualVectorDB(pg_conninfo) as store:
        for strategy in ("token_based", "markdown_aware", "docling_hybrid", "semantic"):
            settings_env(CHUNKING_STRATEGY=strategy)
            summary = ingest_flow(
                str(CORPUS_PATH),
                store=store,
                bm25=bm25_store,
                embeddings=embeddings,
                reingest=True,  # boundaries change but content hashes don't (the v1 gotcha)
                verbose=0,
            )
            assert summary.discovered == 3, strategy  # aurora.md, glossary.md, tidal_grid.txt
            assert summary.ingested == 3, strategy
            assert summary.failed == 0, strategy
            assert summary.chunks > 0, strategy

            with psycopg.connect(pg_conninfo) as conn:
                strategies = conn.execute(
                    "SELECT DISTINCT metadata->>'chunking_strategy' FROM chunks"
                ).fetchall()
                n_breadcrumbs = conn.execute(
                    "SELECT count(*) FROM chunks WHERE metadata->>'heading_path' IS NOT NULL"
                ).fetchone()
            assert strategies == [(strategy,)], strategy  # provenance on every chunk
            assert n_breadcrumbs is not None
            if strategy in heading_aware:  # aurora_station.md is heading-rich
                assert n_breadcrumbs[0] > 0, f"{strategy} stored no heading_path"

            retriever = HybridRetriever(store=store, bm25=bm25_store, embeddings=embeddings)
            llm = ScriptedLLM(SCRIPTED_ANSWER)
            state = query_flow(QUESTION, retriever=retriever, llm=llm, verbose=0)  # type: ignore[arg-type]
            assert any(PLANTED_FACT in chunk.content for chunk in state["retrieved"]), (
                f"{strategy} missed the planted kelp-corridor chunk"
            )
            assert state["answer"] == SCRIPTED_ANSWER


class CondensingAndAnsweringLLM:
    """One fake for both LLM stages, dispatched on the prompt's shape.

    The flow hands a single client to the condense task and the answer
    task (exactly the API's composition), so the fake plays whichever
    role the prompt asks for — recording each, and wrapping the condense
    reply in a ``<think>`` stage to prove stripping end-to-end.
    """

    def __init__(self, standalone: str, answer: str) -> None:
        self.standalone = standalone
        self.answer = answer
        self.condense_prompts: list[str] = []
        self.answer_prompts: list[str] = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
    ) -> str:
        prompt = messages[0]["content"]
        if "STANDALONE QUERY:" in prompt:
            self.condense_prompts.append(prompt)
            return f"<think>they mean the kelp corridor</think>{self.standalone}"
        self.answer_prompts.append(prompt)
        return self.answer


FOLLOW_UP = "How long is it?"


def test_pronoun_follow_up_discriminates_condense_from_simple(
    pinned_settings: None,
    settings_env: Callable[..., None],
    pg_conninfo: str,
    bm25_store: ElasticsearchBM25,
) -> None:
    """★ The chat remembers (spec_v3 §4).

    Turn 2 is a pronoun follow-up whose words alone name nothing: under
    ``simple`` it retrieves the wrong chunks, under ``condense_context``
    the history-resolved rewrite retrieves the planted kelp-corridor fact
    — while the answer prompt still carries the user's own words. This
    asymmetry is the feature's entire reason for existing.
    """
    settings_env(
        CONDENSE_ENABLED="true",
        CONDENSE_MODEL_TYPE="default",
        CONDENSE_HISTORY_TURNS="6",
        CONDENSE_MAX_TOKENS="128",
        CONDENSE_MAX_CHARS="512",
    )
    embeddings = FakeEmbeddings()

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    history = [
        Turn("user", QUESTION),
        Turn("assistant", SCRIPTED_ANSWER),
    ]

    with ContextualVectorDB(pg_conninfo) as store:
        ingest_flow(
            str(CORPUS_PATH), store=store, bm25=bm25_store, embeddings=embeddings, verbose=0
        )
        retriever = HybridRetriever(store=store, bm25=bm25_store, embeddings=embeddings)

        # Under simple, the pronoun query is searched verbatim — and the
        # planted chunk is nowhere in the top-k: the turn 2 words alone
        # cannot find it.
        simple_llm = CondensingAndAnsweringLLM(QUESTION, "I don't know.")
        state = query_flow(
            FOLLOW_UP,
            history=history,
            engine=get_chat_engine("simple"),
            retriever=retriever,
            llm=simple_llm,  # type: ignore[arg-type]
            k=3,
            verbose=0,
        )
        assert state["prepared"].condensed is False
        assert simple_llm.condense_prompts == []  # simple never calls the condenser
        assert embeddings.query_calls[-1] == FOLLOW_UP  # searched verbatim
        assert all(PLANTED_FACT not in chunk.content for chunk in state["retrieved"]), (
            "the fixture stopped discriminating: the bare pronoun query found the fact"
        )

        # Under condense_context, the same turn condenses against history
        # and the rewrite finds the planted fact — ranked first.
        condense_llm = CondensingAndAnsweringLLM(QUESTION, SCRIPTED_ANSWER)
        state = query_flow(
            FOLLOW_UP,
            history=history,
            engine=get_chat_engine("condense_context"),
            retriever=retriever,
            llm=condense_llm,  # type: ignore[arg-type]
            k=3,
            verbose=0,
        )

    assert state["prepared"].condensed is True
    assert state["prepared"].search_query == QUESTION  # think-stripped rewrite
    assert state["prepared"].original_query == FOLLOW_UP
    assert state["prepared"].condense_latency_s is not None

    # The condenser saw the turn-1 exchange, and its rewrite — not the
    # pronoun — drove the query embedding and retrieval.
    assert len(condense_llm.condense_prompts) == 1
    assert QUESTION in condense_llm.condense_prompts[0]
    assert SCRIPTED_ANSWER in condense_llm.condense_prompts[0]
    assert embeddings.query_calls[-1] == QUESTION
    assert any(PLANTED_FACT in chunk.content for chunk in state["retrieved"])
    assert "kelp corridor" in state["retrieved"][0].content

    # The two-string split, end to end: the answer prompt keeps the
    # user's own words, never the rewrite.
    assert len(condense_llm.answer_prompts) == 1
    assert f"QUESTION: {FOLLOW_UP}" in condense_llm.answer_prompts[0]
    assert f"QUESTION: {QUESTION}" not in condense_llm.answer_prompts[0]
    assert state["answer"] == SCRIPTED_ANSWER


def test_all_three_retrieval_methods_answer(
    pinned_settings: None, pg_conninfo: str, bm25_store: ElasticsearchBM25
) -> None:
    """Every RETRIEVAL_METHOD value drives the full Q&A pipeline cleanly."""
    embeddings = FakeEmbeddings()

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    with ContextualVectorDB(pg_conninfo) as store:
        ingest_flow(
            str(CORPUS_PATH), store=store, bm25=bm25_store, embeddings=embeddings, verbose=0
        )
        retrievers = {
            "semantic": SemanticRetriever(store=store, embeddings=embeddings),
            "bm25": BM25Retriever(bm25=bm25_store, store=store),
            "hybrid": HybridRetriever(store=store, bm25=bm25_store, embeddings=embeddings),
        }
        for method, retriever in retrievers.items():
            llm = ScriptedLLM(SCRIPTED_ANSWER)
            state = query_flow(
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
