"""End-to-end walking-skeleton test (Phase 4): ingest → retrieve → answer.

Real pgvector Postgres via testcontainers; deterministic fake embeddings
(hashed bag-of-words, so lexically similar texts land near each other) and a
scripted fake LLM stand in for the GPU services (spec §15.1 "e2e" row).

Select with ``pytest -m e2e`` (needs Docker).
"""

import hashlib
import math
import re
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import psycopg
import pytest
from testcontainers.postgres import PostgresContainer

from varagity.generation.answer import answer_query, format_context
from varagity.ingest.loader import ingest_corpus
from varagity.retrieval.semantic import SemanticRetriever
from varagity.stores.vector_store import ContextualVectorDB

pytestmark = pytest.mark.e2e

SCHEMA_PATH = Path(__file__).parents[2] / "varagity" / "stores" / "schema.sql"
CORPUS_PATH = Path(__file__).parents[1] / "fixtures" / "corpus"
DIM = 1024  # must match the schema's vector(1024)

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


@pytest.fixture
def pinned_settings(settings_env: Callable[..., None]) -> None:
    """Pin every pipeline setting so the machine's .env cannot leak in."""
    settings_env(
        ALLOWED_EXTENSIONS=".pdf,.txt,.md",
        CHUNKING_STRATEGY="recursive_character",
        CHUNK_SIZE=400,
        CHUNK_OVERLAP=50,
        EMBEDDING_MODEL="fake-bow-1024",
        RETRIEVAL_METHOD="semantic",
        TOP_K=10,
    )


def test_walking_skeleton_ingest_to_grounded_answer(
    pinned_settings: None, pg_conninfo: str
) -> None:
    """★ The Phase-4 milestone, automated: fixtures → pgvector → Q&A state."""
    embeddings = FakeEmbeddings()
    llm = ScriptedLLM(f"<think>scanning the context…</think>{SCRIPTED_ANSWER}")

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    with ContextualVectorDB(pg_conninfo) as store:
        summary = ingest_corpus(str(CORPUS_PATH), store=store, embeddings=embeddings, verbose=0)
        assert summary.discovered == 3
        assert summary.ingested == 3
        assert summary.failed == 0
        assert summary.chunks > 0

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


def test_second_run_is_idempotent_and_still_answers(
    pinned_settings: None, pg_conninfo: str
) -> None:
    """Re-running the startup sequence skips unchanged files; Q&A still works."""
    embeddings = FakeEmbeddings()

    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")

    with ContextualVectorDB(pg_conninfo) as store:
        first = ingest_corpus(str(CORPUS_PATH), store=store, embeddings=embeddings, verbose=0)
        second = ingest_corpus(str(CORPUS_PATH), store=store, embeddings=embeddings, verbose=0)
        assert first.ingested == 3
        assert second.ingested == 0
        assert second.skipped == 3

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
