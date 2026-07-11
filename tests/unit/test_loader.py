"""Unit tests for the ingestion loader (identity/contextual paths + guards)."""

import logging
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from varagity.ingest import loader as loader_module
from varagity.ingest.loader import MIN_EXTRACTED_CHARS, ingest_corpus
from varagity.stores.records import ChunkRecord


class FakeStore:
    """In-memory stand-in for ContextualVectorDB."""

    def __init__(self, start_index: int = 0) -> None:
        self.start_index = start_index
        self.documents: dict[str, dict] = {}
        self.records: list[ChunkRecord] = []
        self.embeddings: list[list[float]] = []
        self.closed = False

    def next_original_index(self) -> int:
        return self.start_index

    def document_exists(self, doc_id: str, content_hash: str) -> bool:
        doc = self.documents.get(doc_id)
        return doc is not None and doc["content_hash"] == content_hash

    def document_n_chunks(self, doc_id: str) -> int | None:
        doc = self.documents.get(doc_id)
        return None if doc is None else doc["n_chunks"]

    def upsert_document(
        self, *, doc_id: str, source: str, file_type: str, content_hash: str, n_chunks: int
    ) -> None:
        self.documents[doc_id] = {
            "source": source,
            "file_type": file_type,
            "content_hash": content_hash,
            "n_chunks": n_chunks,
        }

    def upsert_chunks(self, records: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        self.records.extend(records)
        self.embeddings.extend(embeddings)

    def store_document(
        self,
        *,
        doc_id: str,
        source: str,
        file_type: str,
        content_hash: str,
        records: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        self.upsert_document(
            doc_id=doc_id,
            source=source,
            file_type=file_type,
            content_hash=content_hash,
            n_chunks=len(records),
        )
        self.upsert_chunks(records, embeddings)

    def delete_document(self, doc_id: str) -> int:
        if self.documents.pop(doc_id, None) is None:
            return 0
        kept = [
            (r, e) for r, e in zip(self.records, self.embeddings, strict=True) if r.doc_id != doc_id
        ]
        self.records = [r for r, _ in kept]
        self.embeddings = [e for _, e in kept]
        return 1

    def close(self) -> None:
        self.closed = True


class FakeBM25:
    """In-memory stand-in for ElasticsearchBM25 (the dual-write target)."""

    def __init__(self) -> None:
        self.indexed: list[ChunkRecord] = []
        self.create_index_calls = 0
        self.deleted_doc_ids: list[str] = []
        self.closed = False

    def create_index(self) -> bool:
        self.create_index_calls += 1
        return self.create_index_calls == 1

    def index_chunks(self, records: list[ChunkRecord]) -> int:
        self.indexed.extend(records)
        return len(records)

    def delete_document(self, doc_id: str) -> int:
        self.deleted_doc_ids.append(doc_id)
        kept = [record for record in self.indexed if record.doc_id != doc_id]
        deleted = len(self.indexed) - len(kept)
        self.indexed = kept
        return deleted

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def auto_bm25(monkeypatch: pytest.MonkeyPatch) -> list[FakeBM25]:
    """Substitute FakeBM25 for the class the loader constructs when none is injected.

    Keeps every test hermetic (no live Elasticsearch) and returns the
    constructed instances so the owned-store path is assertable.
    """
    created: list[FakeBM25] = []

    class AutoFakeBM25(FakeBM25):
        def __init__(self) -> None:
            super().__init__()
            created.append(self)

    monkeypatch.setattr(loader_module, "ElasticsearchBM25", AutoFakeBM25)
    return created


class FakeEmbeddings:
    """Deterministic stand-in for EmbeddingsClient (passage mode only)."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_passages(self, texts: list[str], verbose: int | None = None) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.25, -0.25, 0.5] for _ in texts]


class FakeContextLLM:
    """Stub LLM for contextualization: records prompts, returns numbered blurbs."""

    def __init__(self, think: bool = False) -> None:
        self.think = think
        self.prompts: list[str] = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
    ) -> str:
        self.prompts.append(messages[0]["content"])
        blurb = f"Situating blurb #{len(self.prompts)}."
        return f"<think>placing the chunk…</think>{blurb}" if self.think else blurb


# The identity path (plan decision #2): pin CONTEXTUALIZE off so these tests
# exercise the non-contextual baseline regardless of the machine's .env.
@pytest.fixture
def pinned_settings(settings_env: Callable[..., None]) -> None:
    settings_env(
        ALLOWED_EXTENSIONS=".pdf,.txt,.md",
        CHUNKING_STRATEGY="recursive_character",
        CHUNK_SIZE=400,
        CHUNK_OVERLAP=50,
        EMBEDDING_MODEL="test-model",
        CONTEXTUALIZE="false",
    )


@pytest.fixture
def contextual_settings(pinned_settings: None, settings_env: Callable[..., None]) -> None:
    settings_env(CONTEXTUALIZE="true")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "aurora.md").write_text(
        "# Aurora\n\nThe Aurora station sits at 2,847 meters and is powered by a "
        "thorium micro-reactor nicknamed Lantern producing 4.2 megawatts at peak."
    )
    (root / "tidal.txt").write_text(
        "The Corvo Tidal Grid consists of 87 underwater turbines in three arrays "
        "named Alba, Bruma, and Cinza, delivering 310 megawatt-hours per day."
    )
    return root


def test_happy_path_skeleton_invariants(pinned_settings: None, corpus: Path) -> None:
    store = FakeStore()
    embeddings = FakeEmbeddings()
    summary = ingest_corpus(str(corpus), store=store, embeddings=embeddings, verbose=0)

    assert summary.discovered == 2
    assert summary.ingested == 2
    assert summary.failed == summary.skipped == summary.no_text == summary.unsupported == 0
    assert summary.chunks == len(store.records) > 0

    for record in store.records:
        assert record.context is None  # identity path: CONTEXTUALIZE off
        assert record.contextualized_content == record.content
        assert record.chunk_id == f"{record.doc_id}::{record.chunk_index}"
        assert record.chunk_size == 400
        assert record.chunking_strategy == "recursive_character"
        assert record.embedding_model == "test-model"
        assert record.extraction == "text"

    # embed_passages received exactly the contextualized (== original) texts
    embedded = [text for call in embeddings.calls for text in call]
    assert embedded == [record.contextualized_content for record in store.records]

    # injected store is NOT closed by the loader (caller owns it)
    assert store.closed is False


def test_original_index_monotonic_from_store_watermark(pinned_settings: None, corpus: Path) -> None:
    store = FakeStore(start_index=5)
    ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
    indexes = [record.original_index for record in store.records]
    assert indexes == list(range(5, 5 + len(indexes)))
    # (doc_id, original_index) unique across the whole run
    assert len({(r.doc_id, r.original_index) for r in store.records}) == len(store.records)


def test_doc_id_is_relative_path_stable(pinned_settings: None, tmp_path: Path) -> None:
    """Same relative layout under two different roots → identical doc_ids."""
    doc_ids = []
    for root_name in ("first-root", "second-root"):
        root = tmp_path / root_name / "docs"
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "a.md").write_text("Identical content, long enough to pass the guard. " * 3)
        store = FakeStore()
        ingest_corpus(str(root), store=store, embeddings=FakeEmbeddings(), verbose=0)
        doc_ids.append(store.records[0].doc_id)
    assert doc_ids[0] == doc_ids[1]


def test_second_run_skips_unchanged_files(pinned_settings: None, corpus: Path) -> None:
    store = FakeStore()
    first = ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
    assert first.ingested == 2

    second = ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
    assert second.ingested == 0
    assert second.skipped == 2
    assert second.chunks == 0
    assert len(store.records) == first.chunks  # nothing re-written


def test_changed_file_is_reingested(pinned_settings: None, corpus: Path) -> None:
    store = FakeStore()
    ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
    (corpus / "aurora.md").write_text(
        "# Aurora, revised\n\nThe station now hosts 16 researchers and the reactor "
        "was upgraded to 4.5 megawatts of peak output for the coming decade."
    )
    summary = ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
    assert summary.ingested == 1  # the changed file
    assert summary.skipped == 1  # the unchanged one


def test_empty_extraction_guard_records_zero_chunk_document(
    pinned_settings: None, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "empty.txt").write_text("   \n\n  \t ")  # below MIN_EXTRACTED_CHARS
    store = FakeStore()
    with caplog.at_level(logging.WARNING):
        summary = ingest_corpus(str(root), store=store, embeddings=FakeEmbeddings(), verbose=0)

    assert summary.no_text == 1
    assert summary.ingested == 0
    assert any("no extractable text" in r.message for r in caplog.records)
    # visibly "seen": a documents row with n_chunks = 0, but no chunks
    assert [doc["n_chunks"] for doc in store.documents.values()] == [0]
    assert store.records == []
    assert MIN_EXTRACTED_CHARS == 50


def test_known_empty_file_rewarns_without_reparsing(
    pinned_settings: None,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "empty.txt").write_text(" \n ")
    store = FakeStore()
    ingest_corpus(str(root), store=store, embeddings=FakeEmbeddings(), verbose=0)

    # Second run: the parser must not run again for the known-empty file.
    real_get_parser = loader_module.get_parser
    extract_calls = []

    def spying_get_parser(name: str):
        parser = real_get_parser(name)

        class Spy:
            def extract(self, path, verbose=None):
                extract_calls.append(path)
                return parser.extract(path, verbose=verbose)

        return Spy()

    monkeypatch.setattr(loader_module, "get_parser", spying_get_parser)
    with caplog.at_level(logging.WARNING):
        summary = ingest_corpus(str(root), store=store, embeddings=FakeEmbeddings(), verbose=0)

    assert summary.no_text == 1  # re-warned, still surfaced in the summary
    assert extract_calls == []  # …but not re-parsed
    assert any("no extractable text" in r.message for r in caplog.records)


def test_pdfs_counted_unsupported_until_phase_7(
    pinned_settings: None, corpus: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (corpus / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    store = FakeStore()
    with caplog.at_level(logging.WARNING):
        summary = ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
    assert summary.discovered == 3
    assert summary.ingested == 2
    assert summary.unsupported == 1
    assert any("no parser registered" in r.message for r in caplog.records)


def test_one_bad_file_does_not_abort_the_run(
    pinned_settings: None, corpus: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (corpus / "broken.txt").write_bytes(b"\xff\xfe\x00 not utf-8 at all \xff" * 20)
    store = FakeStore()
    with caplog.at_level(logging.ERROR):
        summary = ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
    assert summary.failed == 1
    assert summary.ingested == 2  # the good files still landed
    assert any("failed to ingest" in r.message for r in caplog.records)


def test_invalid_verbose_raises(pinned_settings: None, corpus: Path) -> None:
    with pytest.raises(ValueError, match="verbose"):
        ingest_corpus(str(corpus), store=FakeStore(), embeddings=FakeEmbeddings(), verbose=3)


class TestContextualization:
    """The Phase-5 contextual path (spec §9.4; plan decision #2)."""

    def test_blurb_lands_in_context_with_composition(
        self, contextual_settings: None, corpus: Path
    ) -> None:
        store, embeddings = FakeStore(), FakeEmbeddings()
        llm = FakeContextLLM(think=True)
        summary = ingest_corpus(str(corpus), store=store, embeddings=embeddings, llm=llm, verbose=0)

        assert summary.ingested == 2
        assert len(store.records) > 0
        # one LLM call per chunk
        assert len(llm.prompts) == len(store.records)
        for record in store.records:
            assert record.context is not None
            assert record.context.startswith("Situating blurb #")
            assert "<think>" not in record.context  # clean_response applied
            # spec §9.4 composition, via ChunkRecord.create
            assert record.contextualized_content == f"{record.context}\n\n{record.content}"

        # embed_passages received the *contextualized* texts, not the raw chunks
        embedded = [text for call in embeddings.calls for text in call]
        assert embedded == [record.contextualized_content for record in store.records]

    def test_chunks_contextualized_per_document_in_order(
        self, contextual_settings: None, tmp_path: Path
    ) -> None:
        """Stub LLM call order: doc A's chunks (in order), then doc B's.

        Per-document grouping keeps the shared document preamble identical
        across consecutive calls so llama.cpp reuses its prompt cache.
        """
        root = tmp_path / "docs"
        root.mkdir()
        # Each file > CHUNK_SIZE (400 chars) so it splits into ≥ 2 chunks.
        (root / "alpha.txt").write_text(
            "ALPHA-MARKER opening paragraph about the station's aeroponics bay. "
            + "The hydro loops recirculate nutrient film across forty racks. " * 6
        )
        (root / "bravo.txt").write_text(
            "BRAVO-MARKER opening paragraph about the tidal turbine arrays. "
            + "Each turbine reports blade torque to the shore controller. " * 6
        )
        store = FakeStore()
        llm = FakeContextLLM()
        ingest_corpus(str(root), store=store, embeddings=FakeEmbeddings(), llm=llm, verbose=0)

        per_doc = Counter(record.doc_id for record in store.records)
        assert len(per_doc) == 2
        assert all(count >= 2 for count in per_doc.values())

        # Prompt order == record order: per-document grouping and chunk order
        # in one assertion (records accumulate file-by-file, chunk-by-chunk).
        prompt_chunks = [
            prompt.split("<chunk>\n")[1].split("\n</chunk>")[0] for prompt in llm.prompts
        ]
        assert prompt_chunks == [record.content for record in store.records]
        # Every prompt embeds its own parent document, never the other one.
        for prompt, record in zip(llm.prompts, store.records, strict=True):
            marker = "ALPHA-MARKER" if record.file_name == "alpha.txt" else "BRAVO-MARKER"
            other = "BRAVO-MARKER" if marker == "ALPHA-MARKER" else "ALPHA-MARKER"
            document = prompt.split("</document>")[0]
            assert marker in document
            assert other not in document

    def test_contextualize_off_keeps_identity_and_never_calls_llm(
        self, pinned_settings: None, corpus: Path
    ) -> None:
        store = FakeStore()
        llm = FakeContextLLM()
        summary = ingest_corpus(
            str(corpus), store=store, embeddings=FakeEmbeddings(), llm=llm, verbose=0
        )
        assert summary.ingested == 2
        assert llm.prompts == []  # the injected client is ignored when off
        for record in store.records:
            assert record.context is None
            assert record.contextualized_content == record.content


class TestReingest:
    """`ingest --reingest`: re-process unchanged files after setting changes."""

    def test_unchanged_files_are_reprocessed_without_duplicates(
        self, pinned_settings: None, corpus: Path
    ) -> None:
        store = FakeStore()
        first = ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
        again = ingest_corpus(
            str(corpus), store=store, embeddings=FakeEmbeddings(), reingest=True, verbose=0
        )
        assert again.ingested == 2
        assert again.skipped == 0
        # previous ingests were deleted first — no duplicate chunks
        assert len(store.records) == first.chunks
        assert len(store.documents) == 2

    def test_toggling_contextualize_alone_skips_unchanged_files(
        self, pinned_settings: None, settings_env: Callable[..., None], corpus: Path
    ) -> None:
        """The documented gotcha: config changes don't change content hashes."""
        store = FakeStore()
        ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)

        settings_env(CONTEXTUALIZE="true")
        summary = ingest_corpus(
            str(corpus), store=store, embeddings=FakeEmbeddings(), llm=FakeContextLLM(), verbose=0
        )
        assert summary.skipped == 2  # unchanged bytes → skipped despite the toggle
        assert all(record.context is None for record in store.records)

    def test_reingest_upgrades_identity_ingest_to_contextual(
        self, pinned_settings: None, settings_env: Callable[..., None], corpus: Path
    ) -> None:
        """The Phase-5 migration path: baseline corpus → --reingest → blurbs."""
        store = FakeStore()
        ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
        assert all(record.context is None for record in store.records)

        settings_env(CONTEXTUALIZE="true")
        summary = ingest_corpus(
            str(corpus),
            store=store,
            embeddings=FakeEmbeddings(),
            llm=FakeContextLLM(),
            reingest=True,
            verbose=0,
        )
        assert summary.ingested == 2
        assert store.records  # rewritten…
        assert all(record.context for record in store.records)  # …with blurbs


class TestDualWrite:
    """Spec §9.6 (Phase 6): every chunk lands in both stores, or the file fails."""

    def test_both_stores_receive_the_same_chunks(self, pinned_settings: None, corpus: Path) -> None:
        store, bm25 = FakeStore(), FakeBM25()
        summary = ingest_corpus(
            str(corpus), store=store, bm25=bm25, embeddings=FakeEmbeddings(), verbose=0
        )
        assert summary.ingested == 2
        assert bm25.create_index_calls == 1  # idempotent create at run start
        assert [r.chunk_id for r in bm25.indexed] == [r.chunk_id for r in store.records]
        assert [r.contextualized_content for r in bm25.indexed] == [
            r.contextualized_content for r in store.records
        ]
        # injected BM25 store is NOT closed by the loader (caller owns it)
        assert bm25.closed is False

    def test_owned_bm25_store_constructed_and_closed(
        self, pinned_settings: None, corpus: Path, auto_bm25: list[FakeBM25]
    ) -> None:
        store = FakeStore()
        ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
        assert len(auto_bm25) == 1  # constructed from settings…
        assert auto_bm25[0].closed is True  # …and closed on return
        assert [r.chunk_id for r in auto_bm25[0].indexed] == [r.chunk_id for r in store.records]

    def test_es_failure_fails_file_before_pg_commit(
        self, pinned_settings: None, corpus: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """BM25 is written first: a failed file leaves no idempotency marker.

        If the pgvector ``documents`` row landed before the Elasticsearch
        failure, the next run would skip the file and the stores would stay
        inconsistent forever.
        """

        class ExplodingBM25(FakeBM25):
            def index_chunks(self, records: list[ChunkRecord]) -> int:
                raise RuntimeError("elasticsearch exploded mid-bulk")

        store, bm25 = FakeStore(), ExplodingBM25()
        with caplog.at_level(logging.ERROR):
            summary = ingest_corpus(
                str(corpus), store=store, bm25=bm25, embeddings=FakeEmbeddings(), verbose=0
            )
        assert summary.failed == 2
        assert summary.ingested == 0
        assert store.documents == {}  # no marker → both files re-attempted next run
        assert store.records == []
        assert any("failed to ingest" in r.message for r in caplog.records)

    def test_empty_file_writes_nothing_to_es(self, pinned_settings: None, tmp_path: Path) -> None:
        root = tmp_path / "docs"
        root.mkdir()
        (root / "empty.txt").write_text(" \n ")
        bm25 = FakeBM25()
        summary = ingest_corpus(
            str(root), store=FakeStore(), bm25=bm25, embeddings=FakeEmbeddings(), verbose=0
        )
        assert summary.no_text == 1
        assert bm25.indexed == []  # the 0-chunk documents row is pgvector-only

    def test_reingest_deletes_from_both_stores(self, pinned_settings: None, corpus: Path) -> None:
        store, bm25 = FakeStore(), FakeBM25()
        first = ingest_corpus(
            str(corpus), store=store, bm25=bm25, embeddings=FakeEmbeddings(), verbose=0
        )
        again = ingest_corpus(
            str(corpus),
            store=store,
            bm25=bm25,
            embeddings=FakeEmbeddings(),
            reingest=True,
            verbose=0,
        )
        assert again.ingested == 2
        assert sorted(bm25.deleted_doc_ids) == sorted(store.documents)
        assert len(bm25.indexed) == first.chunks  # re-indexed without duplicates
