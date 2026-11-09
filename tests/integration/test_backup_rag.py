"""Integration tests for the RAG export and restore against real stores (ADR-019).

Runs the real ``schema.sql`` in a throwaway ``pgvector/pgvector:pg16``
container and a throwaway single-node Elasticsearch (the
:mod:`varagity.eval.containers` helpers), seeds documents and chunks through
the ingest's own write paths, and exports them: the bundle's members must
match the rows, and every vector must come back **bit-exact** (the snapshot
reads pgvector's binary format; nothing round-trips through decimal text).

Then the restore: export → wipe → restore must give back the same rows
(``fetch_by_identity``) and the same BM25 hits, with the vector index
rebuilt; a ``COPY`` that fails mid-stream must leave the corpus it was
replacing exactly as it was; and the corpus's advisory lock must keep a
restore and an ingest apart across two database sessions.

Select with ``pytest -m integration`` (needs Docker).
"""

import hashlib
import json
import random
import struct
import zipfile
from array import array
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest
from elasticsearch import Elasticsearch

from varagity.backup.bundle import MANIFEST_NAME, BundleInvalid, read_manifest
from varagity.backup.rag import (
    CHUNKS_MEMBER,
    DOCUMENTS_MEMBER,
    EMBEDDINGS_MEMBER,
    export_rag,
    restore_rag,
)
from varagity.eval.containers import ephemeral_elasticsearch, ephemeral_postgres
from varagity.ingest.loader import ingest_corpus
from varagity.stores.bm25_store import ElasticsearchBM25
from varagity.stores.records import ChunkRecord
from varagity.stores.vector_store import ContextualVectorDB, CorpusLocked

pytestmark = pytest.mark.integration

DIM = 1024
SETTINGS = {"CHUNK_SIZE": "400", "CONTEXTUALIZE": "true"}
INDEX = "varagity_test_restore_bm25"

# float32 edge values a decimal round trip is most likely to bend: the
# smallest subnormal, the largest finite value, and an ordinary fraction.
EDGES = [1.401298464324817e-45, 3.4028234663852886e38, 0.1]


@pytest.fixture(scope="module")
def pg_conninfo() -> Iterator[str]:
    """A pgvector Postgres with schema.sql applied, for this module."""
    with ephemeral_postgres() as conninfo:
        yield conninfo


@pytest.fixture
def store(pg_conninfo: str) -> Iterator[ContextualVectorDB]:
    """A store on clean tables (truncated per test)."""
    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")
    with ContextualVectorDB(pg_conninfo) as db:
        yield db


@pytest.fixture(scope="module")
def es_url() -> Iterator[str]:
    """A single-node Elasticsearch for this module."""
    with ephemeral_elasticsearch() as url:
        yield url


@pytest.fixture
def bm25(es_url: str) -> Iterator[ElasticsearchBM25]:
    """A BM25 store on a fresh index (deleted and recreated per test)."""
    raw = Elasticsearch(es_url)
    raw.indices.delete(index=INDEX, ignore_unavailable=True)
    raw.close()
    with ElasticsearchBM25(url=es_url, index_name=INDEX) as store:
        store.create_index()
        yield store


@pytest.fixture
def docs_root(tmp_path: Path, settings_env: Callable[..., None]) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    settings_env(DOCS_PATH=str(root), EMBEDDING_MODEL="test-model", EMBEDDING_DIM=DIM)
    return root


def float32_vector(rng: random.Random) -> list[float]:
    """A random vector whose values are exactly representable as float32."""
    return array("f", (rng.uniform(-1.0, 1.0) for _ in range(DIM))).tolist()


def seed(
    store: ContextualVectorDB,
    doc_id: str,
    source: str,
    vectors: list[list[float]],
    start: int,
    *,
    bm25: ElasticsearchBM25 | None = None,
    texts: list[str] | None = None,
) -> None:
    """Write one document through the ingest's own store paths (BM25 first, like ingest)."""
    records = [
        ChunkRecord.create(
            doc_id=doc_id,
            original_index=start + index,
            chunk_index=index,
            source=source,
            file_name=Path(source).name,
            file_type="md",
            page=None,
            content=texts[index] if texts else f"{doc_id} chunk {index} — ünïcode",
            context=None if index % 2 else f"blurb for chunk {index}",
            chunk_size=400,
            chunk_overlap=50,
            chunking_strategy="recursive_character",
            embedding_model="test-model",
            content_hash="hash-" + doc_id,
        )
        for index in range(len(vectors))
    ]
    if bm25 is not None:
        bm25.index_chunks(records)
    store.store_document(
        doc_id=doc_id,
        source=source,
        file_type="md",
        content_hash="hash-" + doc_id,
        records=records,
        embeddings=vectors,
    )


def members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode().splitlines()]


class TestRagExport:
    def test_the_bundle_matches_the_rows_and_the_vectors_are_bit_exact(
        self, store: ContextualVectorDB, docs_root: Path, pg_conninfo: str
    ) -> None:
        rng = random.Random(19)
        nested = [float32_vector(rng) for _ in range(3)]
        nested[0][:3] = EDGES
        flat = [float32_vector(rng) for _ in range(2)]
        outside = [float32_vector(rng)]
        # Seeded out of ingest order on purpose: the bundle follows
        # original_index, not insertion order.
        seed(store, "docflat000000000", str(docs_root / "b.md"), flat, start=3)
        seed(store, "docnest000000000", str(docs_root / "q3" / "a.md"), nested, start=0)
        seed(store, "docoutside000000", "/somewhere/else/c.md", outside, start=5)
        store.upsert_document(
            doc_id="docempty00000000",
            source=str(docs_root / "scan.pdf"),
            file_type="pdf",
            content_hash="hash-empty",
            n_chunks=0,
        )

        result = export_rag(docs_root / "varagity-exports", store, SETTINGS, stale=False)

        manifest = read_manifest(result.bundle.path)
        assert manifest.counts == {"documents": 4, "chunks": 6}
        assert manifest.embedding_dim == DIM
        bundle = members(result.bundle.path)

        # documents.jsonl: every column of every row, plus the relative path.
        documents = {entry["doc_id"]: entry for entry in jsonl(bundle[DOCUMENTS_MEMBER])}
        with psycopg.connect(pg_conninfo) as conn:
            rows = conn.execute(
                "SELECT doc_id, source, file_type, content_hash, n_chunks, ingested_at "
                "FROM documents"
            ).fetchall()
        assert set(documents) == {row[0] for row in rows}
        for doc_id, source, file_type, content_hash, n_chunks, ingested_at in rows:
            entry = documents[doc_id]
            assert entry["source"] == source
            assert entry["file_type"] == file_type
            assert entry["content_hash"] == content_hash
            assert entry["n_chunks"] == n_chunks
            assert datetime.fromisoformat(entry["ingested_at"]) == ingested_at
        assert documents["docnest000000000"]["relative_path"] == "q3/a.md"
        assert documents["docflat000000000"]["relative_path"] == "b.md"
        assert documents["docempty00000000"]["relative_path"] == "scan.pdf"
        assert documents["docoutside000000"]["relative_path"] is None

        # chunks.jsonl: every column but the vector, original_index order.
        lines = jsonl(bundle[CHUNKS_MEMBER])
        with psycopg.connect(pg_conninfo) as conn:
            chunk_rows = conn.execute(
                "SELECT chunk_id, doc_id, original_index, chunk_index, content, context, "
                "contextualized_content, metadata, created_at FROM chunks "
                "ORDER BY original_index, doc_id"
            ).fetchall()
        assert [line["chunk_id"] for line in lines] == [row[0] for row in chunk_rows]
        assert [line["original_index"] for line in lines] == list(range(6))
        for line, row in zip(lines, chunk_rows, strict=True):
            assert line["doc_id"] == row[1]
            assert line["chunk_index"] == row[3]
            assert line["content"] == row[4]
            assert line["context"] == row[5]
            assert line["contextualized_content"] == row[6]
            assert line["metadata"] == row[7]
            assert datetime.fromisoformat(line["created_at"]) == row[8]
            assert "embedding" not in line

        # embeddings.f32: row i is line i's vector, to the bit.
        seeded = {
            "docnest000000000": nested,
            "docflat000000000": flat,
            "docoutside000000": outside,
        }
        matrix = bundle[EMBEDDINGS_MEMBER]
        assert len(matrix) == len(lines) * DIM * 4
        for index, line in enumerate(lines):
            expected = struct.pack(f"<{DIM}f", *seeded[line["doc_id"]][line["chunk_index"]])
            assert matrix[index * DIM * 4 : (index + 1) * DIM * 4] == expected, line["chunk_id"]

    def test_a_write_committed_mid_export_is_not_in_the_snapshot(
        self, store: ContextualVectorDB, docs_root: Path, pg_conninfo: str
    ) -> None:
        """REPEATABLE READ: the counts and both row streams are one moment."""
        rng = random.Random(23)
        seed(store, "docbefore0000000", str(docs_root / "a.md"), [float32_vector(rng)], start=0)
        with store.iter_corpus() as snapshot:
            with ContextualVectorDB(pg_conninfo) as writer:
                seed(writer, "docafter00000000", str(docs_root / "b.md"), [float32_vector(rng)], 1)
            assert (snapshot.document_count, snapshot.chunk_count) == (1, 1)
            assert [row.doc_id for row in snapshot.documents()] == ["docbefore0000000"]
            assert [row.chunk_id for row in snapshot.chunks()] == ["docbefore0000000::0"]
        assert store.document_count() == 2  # the write did commit — just not into the snapshot


# ── restore ───────────────────────────────────────────────────────────────

KELP = ["the kelp corridor runs nine kilometres", "lantern feeds the aurora station"]
TIDAL = ["the tidal grid has eighty-seven turbines", "alba bruma and cinza are its arrays"]


def seed_corpus(
    store: ContextualVectorDB, bm25: ElasticsearchBM25, docs_root: Path, rng: random.Random
) -> dict[str, list[list[float]]]:
    """A small corpus in both stores — one nested doc, one flat, one from outside the root."""
    vectors = {
        "dockelp000000000": [float32_vector(rng) for _ in KELP],
        "doctidal00000000": [float32_vector(rng) for _ in TIDAL],
        "docoutside000000": [float32_vector(rng)],
    }
    vectors["dockelp000000000"][0][:3] = EDGES
    seed(
        store,
        "dockelp000000000",
        str(docs_root / "q3" / "kelp.md"),
        vectors["dockelp000000000"],
        0,
        bm25=bm25,
        texts=KELP,
    )
    seed(
        store,
        "doctidal00000000",
        str(docs_root / "tidal.md"),
        vectors["doctidal00000000"],
        2,
        bm25=bm25,
        texts=TIDAL,
    )
    seed(
        store,
        "docoutside000000",
        "/somewhere/else/c.md",
        vectors["docoutside000000"],
        4,
        bm25=bm25,
        texts=["an outsider"],
    )
    return vectors


def wipe(pg_conninfo: str, es_url: str) -> None:
    """What ``docker compose down -v`` leaves: empty tables, no BM25 index."""
    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")
    raw = Elasticsearch(es_url)
    raw.indices.delete(index=INDEX, ignore_unavailable=True)
    raw.close()


def all_identities(pg_conninfo: str) -> list[tuple[str, int]]:
    with psycopg.connect(pg_conninfo) as conn:
        rows = conn.execute(
            "SELECT doc_id, original_index FROM chunks ORDER BY original_index"
        ).fetchall()
    return [(doc_id, int(index)) for doc_id, index in rows]


def hits(bm25: ElasticsearchBM25, query: str) -> list[tuple[str, int, float]]:
    return [
        (hit.doc_id, hit.original_index, round(hit.score, 4))
        for hit in bm25.search(query, k=10, verbose=0)
    ]


def stored_vectors(store: ContextualVectorDB) -> dict[str, bytes]:
    with store.iter_corpus() as snapshot:
        return {row.chunk_id: row.embedding.to_binary() for row in snapshot.chunks()}


def hnsw_definition(pg_conninfo: str) -> str | None:
    with psycopg.connect(pg_conninfo) as conn:
        row = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_hnsw'"
        ).fetchone()
    return None if row is None else str(row[0])


def resealed(path: Path, replace: dict[str, bytes]) -> Path:
    """A self-consistent copy of a bundle with some members replaced (sha256s re-recorded)."""
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    contents.update(replace)
    manifest = json.loads(contents[MANIFEST_NAME])
    for member in manifest["members"]:
        data = contents[member["name"]]
        member["size_bytes"] = len(data)
        member["sha256"] = hashlib.sha256(data).hexdigest()
    contents[MANIFEST_NAME] = json.dumps(manifest).encode()
    target = path.with_name(f"resealed-{path.name}")
    with zipfile.ZipFile(target, "w") as archive:
        for name, data in contents.items():
            if name != MANIFEST_NAME:
                archive.writestr(name, data)
        archive.writestr(MANIFEST_NAME, contents[MANIFEST_NAME])
    return target


class TestRagRestore:
    def test_export_wipe_restore_gives_back_the_same_rows_and_bm25_hits(
        self,
        store: ContextualVectorDB,
        bm25: ElasticsearchBM25,
        docs_root: Path,
        pg_conninfo: str,
        es_url: str,
    ) -> None:
        """★ The ticket's acceptance shape, in miniature: no model call, nothing lost."""
        vectors = seed_corpus(store, bm25, docs_root, random.Random(29))
        keys = all_identities(pg_conninfo)
        before_rows = store.fetch_by_identity(keys)
        before_hits = {query: hits(bm25, query) for query in ("kelp corridor", "tidal turbines")}
        before_vectors = stored_vectors(store)
        before_index = hnsw_definition(pg_conninfo)
        bundle = export_rag(
            docs_root / "varagity-exports", store, SETTINGS, stale=False
        ).bundle.path

        wipe(pg_conninfo, es_url)
        assert store.document_count() == 0

        frames: list[str] = []
        result = restore_rag(
            bundle,
            store,
            bm25,
            docs_root,
            on_progress=lambda stage, current, total: frames.append(stage),
        )

        assert result.counts == {"documents": 3, "chunks": 5, "bm25": 5}
        after_rows = store.fetch_by_identity(keys)
        assert set(after_rows) == set(before_rows)
        for key, row in before_rows.items():
            restored = after_rows[key]
            assert restored.chunk_id == row.chunk_id
            assert restored.content == row.content
            assert restored.context == row.context
            # Restored under the same DOCS_PATH: metadata (source included) is identical.
            assert restored.metadata == row.metadata
        assert stored_vectors(store) == before_vectors  # bit-exact through binary COPY
        assert {query: hits(bm25, query) for query in before_hits} == before_hits
        assert hits(bm25, "kelp corridor")[0][:2] == ("dockelp000000000", 0)
        assert hnsw_definition(pg_conninfo) == before_index  # the vector index is back
        assert frames[0] == "validate" and frames[-1] == "bm25"
        # The semantic arm answers from the restored vectors: a chunk's own
        # vector finds the chunk.
        (nearest,) = store.search(vectors["doctidal00000000"][0], 1, verbose=0)
        assert nearest.chunk_id == "doctidal00000000::0"
        assert nearest.score == pytest.approx(1.0, abs=1e-5)

    def test_sources_are_re_rooted_onto_the_restoring_corpus_directory(
        self,
        store: ContextualVectorDB,
        bm25: ElasticsearchBM25,
        docs_root: Path,
        tmp_path: Path,
        pg_conninfo: str,
        es_url: str,
    ) -> None:
        seed_corpus(store, bm25, docs_root, random.Random(31))
        bundle = export_rag(
            docs_root / "varagity-exports", store, SETTINGS, stale=False
        ).bundle.path
        wipe(pg_conninfo, es_url)
        other_root = tmp_path / "host-docs"
        restore_rag(bundle, store, bm25, other_root)
        sources = {info.doc_id: info.source for info in store.list_documents()}
        assert sources["dockelp000000000"] == str((other_root / "q3" / "kelp.md").resolve())
        assert sources["docoutside000000"] == "/somewhere/else/c.md"
        with psycopg.connect(pg_conninfo) as conn:
            chunk_sources = dict(
                conn.execute("SELECT chunk_id, metadata->>'source' FROM chunks").fetchall()
            )
        assert chunk_sources["dockelp000000000::1"] == sources["dockelp000000000"]

    @pytest.mark.parametrize("failure", ["malformed_line", "duplicate_identity"])
    def test_a_copy_that_fails_mid_stream_leaves_the_old_corpus_intact(
        self,
        store: ContextualVectorDB,
        bm25: ElasticsearchBM25,
        docs_root: Path,
        pg_conninfo: str,
        failure: str,
    ) -> None:
        """★ Client-side (a bad line) or server-side (a unique violation): one rollback."""
        seed_corpus(store, bm25, docs_root, random.Random(37))
        bundle = export_rag(
            docs_root / "varagity-exports", store, SETTINGS, stale=False
        ).bundle.path
        with zipfile.ZipFile(bundle) as archive:
            lines = archive.read(CHUNKS_MEMBER).splitlines(keepends=True)
        if failure == "malformed_line":
            lines[3] = b'{"chunk_id": "x", "this is": "not a chunk row"}\n'
        else:  # line 2 claims line 1's (doc_id, original_index) under its own chunk_id
            second = json.loads(lines[1])
            second["original_index"] = json.loads(lines[0])["original_index"]
            lines[1] = (json.dumps(second) + "\n").encode()
        broken = resealed(bundle, {CHUNKS_MEMBER: b"".join(lines)})

        # The corpus the restore would replace: a different, current one.
        with psycopg.connect(pg_conninfo, autocommit=True) as conn:
            conn.execute("TRUNCATE documents CASCADE")
        rng = random.Random(41)
        seed(store, "doccurrent000000", str(docs_root / "now.md"), [float32_vector(rng)], 0)
        index_before = hnsw_definition(pg_conninfo)

        expected = BundleInvalid if failure == "malformed_line" else psycopg.errors.UniqueViolation
        with pytest.raises(expected):
            restore_rag(broken, store, bm25, docs_root)

        assert [info.doc_id for info in store.list_documents()] == ["doccurrent000000"]
        assert store.chunk_count() == 1
        assert hnsw_definition(pg_conninfo) == index_before  # the DROP rolled back too
        assert store.corpus_lock_free()  # and the lock was given back

    def test_a_tuned_vector_index_comes_back_as_it_was(
        self,
        store: ContextualVectorDB,
        bm25: ElasticsearchBM25,
        docs_root: Path,
        pg_conninfo: str,
    ) -> None:
        with psycopg.connect(pg_conninfo, autocommit=True) as conn:
            conn.execute("DROP INDEX chunks_embedding_hnsw")
            conn.execute(
                "CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw "
                "(embedding vector_cosine_ops) WITH (m = 8)"
            )
        try:
            seed_corpus(store, bm25, docs_root, random.Random(43))
            bundle = export_rag(
                docs_root / "varagity-exports", store, SETTINGS, stale=False
            ).bundle.path
            restore_rag(bundle, store, bm25, docs_root)
            definition = hnsw_definition(pg_conninfo)
            assert definition is not None and "m='8'" in definition
        finally:
            with psycopg.connect(pg_conninfo, autocommit=True) as conn:
                conn.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw")
                conn.execute(
                    "CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw "
                    "(embedding vector_cosine_ops)"
                )


class TestCorpusLock:
    """Two sessions, one advisory key: what keeps a CLI ingest and a restore apart."""

    def test_a_held_shared_lock_refuses_a_restore(
        self,
        store: ContextualVectorDB,
        bm25: ElasticsearchBM25,
        docs_root: Path,
        pg_conninfo: str,
    ) -> None:
        seed_corpus(store, bm25, docs_root, random.Random(47))
        bundle = export_rag(
            docs_root / "varagity-exports", store, SETTINGS, stale=False
        ).bundle.path
        with (
            ContextualVectorDB(pg_conninfo) as ingest_session,
            ingest_session.corpus_lock(exclusive=False),
        ):
            assert store.corpus_lock_free() is False
            with pytest.raises(CorpusLocked, match="an ingest is running in another process"):
                restore_rag(bundle, store, bm25, docs_root)
            # Shared holds overlap: a second ingest is still welcome.
            with store.corpus_lock(exclusive=False):
                pass
        assert store.corpus_lock_free() is True
        assert store.document_count() == 3  # the refused restore touched nothing

    def test_a_held_exclusive_lock_refuses_an_ingest(
        self,
        store: ContextualVectorDB,
        bm25: ElasticsearchBM25,
        docs_root: Path,
        pg_conninfo: str,
    ) -> None:
        (docs_root / "new.md").write_text(
            "A brand new document, long enough to clear the guard. " * 3
        )
        with (
            ContextualVectorDB(pg_conninfo) as restore_session,
            restore_session.corpus_lock(exclusive=True),
            pytest.raises(CorpusLocked, match="the corpus is being restored"),
        ):
            ingest_corpus(
                str(docs_root),
                store=store,
                bm25=bm25,
                embeddings=object(),  # type: ignore[arg-type]
                verbose=0,
            )
        assert store.document_count() == 0  # refused before its first write

    def test_a_crashed_holder_releases_the_lock_with_its_session(self, pg_conninfo: str) -> None:
        holder = ContextualVectorDB(pg_conninfo)
        with ContextualVectorDB(pg_conninfo) as other:
            lock = holder.corpus_lock(exclusive=False)
            lock.__enter__()
            assert other.corpus_lock_free() is False
            holder.close()  # the process dies without unlocking
            assert other.corpus_lock_free() is True
