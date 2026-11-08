"""Integration tests for the RAG export against a real pgvector Postgres (ADR-019).

Runs the real ``schema.sql`` in a throwaway ``pgvector/pgvector:pg16``
container (the :mod:`varagity.eval.containers` helpers), seeds documents and
chunks through the ingest's own write path, and exports them: the bundle's
members must match the rows, and every vector must come back **bit-exact**
(the snapshot reads pgvector's binary format; nothing round-trips through
decimal text). Phase 2 extends this file into the export → wipe → restore
round trip.

Select with ``pytest -m integration`` (needs Docker).
"""

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

from varagity.backup.bundle import read_manifest
from varagity.backup.rag import CHUNKS_MEMBER, DOCUMENTS_MEMBER, EMBEDDINGS_MEMBER, export_rag
from varagity.eval.containers import ephemeral_postgres
from varagity.stores.records import ChunkRecord
from varagity.stores.vector_store import ContextualVectorDB

pytestmark = pytest.mark.integration

DIM = 1024
SETTINGS = {"CHUNK_SIZE": "400", "CONTEXTUALIZE": "true"}

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
    store: ContextualVectorDB, doc_id: str, source: str, vectors: list[list[float]], start: int
) -> None:
    records = [
        ChunkRecord.create(
            doc_id=doc_id,
            original_index=start + index,
            chunk_index=index,
            source=source,
            file_name=Path(source).name,
            file_type="md",
            page=None,
            content=f"{doc_id} chunk {index} — ünïcode",
            context=None if index % 2 else f"blurb for chunk {index}",
            chunk_size=400,
            chunk_overlap=50,
            chunking_strategy="recursive_character",
            embedding_model="test-model",
            content_hash="hash-" + doc_id,
        )
        for index in range(len(vectors))
    ]
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
