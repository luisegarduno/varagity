"""Integration tests for the Phase 8 corpus + settings surfaces.

The real app over httpx's ASGI transport, pointed at real testcontainers
(pgvector Postgres + single-node Elasticsearch) via the same ``settings_env``
mechanism host-mode runs use. Covers the plan's Phase 8 integration
criteria: ``PATCH /api/settings`` persists + clears the cache + flags
stale (surviving a simulated api restart), migration ``002`` idempotency
rides the runner suite, document upload validation over the real multipart
path, and document delete removing chunks from **both** stores.

Select with ``pytest -m integration`` (needs Docker).
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
from fastapi import FastAPI

from varagity.api import runtime_settings
from varagity.api.main import create_app
from varagity.config import get_settings
from varagity.eval.containers import EphemeralStores, ephemeral_stores
from varagity.stores.app_settings_store import AppSettingsStore
from varagity.stores.migrate import run_migrations
from varagity.stores.records import ChunkRecord, content_hash

pytestmark = pytest.mark.integration

INDEX_NAME = "varagity_corpus_api_test"
EMBEDDING_DIM = 1024


@pytest.fixture(scope="module")
def stores() -> Iterator[EphemeralStores]:
    """Both throwaway containers, migrated, for the whole module."""
    with ephemeral_stores(index_name=INDEX_NAME) as handles:
        with psycopg.connect(handles.pg_conninfo, autocommit=True) as conn:
            run_migrations(conn)
        yield handles


@pytest.fixture
def app(
    stores: EphemeralStores,
    settings_env: Callable[..., None],
    tmp_path: Path,
) -> Iterator[FastAPI]:
    """The real app pointed at the containers, with clean tables + env."""
    parts = psycopg.conninfo.conninfo_to_dict(stores.pg_conninfo)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    settings_env(
        POSTGRES_HOST=parts["host"],
        POSTGRES_PORT=parts["port"],
        POSTGRES_DB=parts["dbname"],
        POSTGRES_USER=parts["user"],
        POSTGRES_PASSWORD=parts["password"],
        ELASTICSEARCH_URL=stores.es_url,
        BM25_INDEX_NAME=INDEX_NAME,
        DOCS_PATH=str(docs_root),
        UPLOAD_MAX_MB=1,
    )
    with psycopg.connect(stores.pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE app_settings")
        conn.execute("TRUNCATE documents CASCADE")
    runtime_settings.reset_for_tests()
    yield create_app()
    runtime_settings.reset_for_tests()


async def request(app: FastAPI, method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        return await client.request(method, path, **kwargs)


def make_records(doc_id: str, source: str, texts: list[str], start_index: int) -> list[ChunkRecord]:
    return [
        ChunkRecord.create(
            doc_id=doc_id,
            original_index=start_index + i,
            chunk_index=i,
            source=source,
            file_name=Path(source).name,
            file_type="txt",
            page=None,
            extraction="text",
            heading_path=None,
            content=text,
            context=None,
            chunk_size=400,
            chunk_overlap=50,
            chunking_strategy="recursive_character",
            embedding_model="test-model",
            content_hash=content_hash(source.encode()),
        )
        for i, text in enumerate(texts)
    ]


def seed_document(
    stores: EphemeralStores, doc_id: str, source: str, texts: list[str], start_index: int
) -> None:
    """Land one document's chunks in both stores, exactly like an ingest."""
    records = make_records(doc_id, source, texts, start_index)
    stores.bm25.create_index()
    stores.bm25.index_chunks(records)
    stores.store.store_document(
        doc_id=doc_id,
        source=source,
        file_type="txt",
        content_hash=records[0].content_hash,
        records=records,
        embeddings=[[0.0] * EMBEDDING_DIM for _ in records],
    )


class TestSettingsPersistence:
    async def test_patch_persists_applies_and_survives_restart(
        self, app: FastAPI, stores: EphemeralStores
    ) -> None:
        base = get_settings().TOP_K
        response = await request(
            app, "PATCH", "/api/settings", json={"overrides": {"TOP_K": base + 21}}
        )
        assert response.status_code == 200

        # Cache cleared: the process-wide settings reflect it immediately.
        assert base + 21 == get_settings().TOP_K
        # Persisted: the row is in the real table.
        with psycopg.connect(stores.pg_conninfo) as conn:
            rows = dict(conn.execute("SELECT key, value FROM app_settings").fetchall())
        assert rows == {"TOP_K": str(base + 21)}

        # Simulated api restart: fresh process state + the lifespan replay.
        runtime_settings.reset_for_tests()
        assert base == get_settings().TOP_K
        with AppSettingsStore(stores.pg_conninfo) as settings_store:
            runtime_settings.load_persisted_overrides(settings_store.load_overrides)
        assert base + 21 == get_settings().TOP_K

    async def test_reingest_affecting_patch_flags_stale_in_the_table(
        self, app: FastAPI, stores: EphemeralStores
    ) -> None:
        seed_document(stores, "docA", "/docs/a.txt", ["alpha content"], start_index=0)
        target = "token_based" if get_settings().CHUNKING_STRATEGY != "token_based" else "semantic"
        response = await request(
            app, "PATCH", "/api/settings", json={"overrides": {"CHUNKING_STRATEGY": target}}
        )
        assert response.status_code == 200
        assert response.json()["corpus_stale"] is True
        with AppSettingsStore(stores.pg_conninfo) as settings_store:
            assert settings_store.is_corpus_stale() is True
        # And GET keeps reporting it (the persistent affordance).
        get_response = await request(app, "GET", "/api/settings")
        assert get_response.json()["corpus_stale"] is True

    async def test_invalid_patch_persists_nothing(
        self, app: FastAPI, stores: EphemeralStores
    ) -> None:
        response = await request(
            app, "PATCH", "/api/settings", json={"overrides": {"SEMANTIC_WEIGHT": 0.9}}
        )
        assert response.status_code == 422  # weight pair must sum to 1.0
        with psycopg.connect(stores.pg_conninfo) as conn:
            assert conn.execute("SELECT count(*) FROM app_settings").fetchone()[0] == 0


class TestUploadValidation:
    async def test_extension_and_size_enforced_on_the_real_multipart_path(
        self, app: FastAPI
    ) -> None:
        docs_root = Path(get_settings().DOCS_PATH)
        files = [
            ("files", ("ok.md", b"# fine", "text/markdown")),
            ("files", ("nope.exe", b"binary", "application/octet-stream")),
            ("files", ("big.txt", b"x" * (1024 * 1024 + 1), "text/plain")),
        ]
        response = await request(app, "POST", "/api/documents", files=files)
        assert response.status_code == 201
        outcomes = {entry["file_name"]: entry for entry in response.json()["files"]}
        assert outcomes["ok.md"]["stored"] is True
        assert outcomes["nope.exe"]["reason"] == "extension_not_allowed"
        assert outcomes["big.txt"]["reason"] == "file_too_large"
        assert sorted(p.name for p in docs_root.iterdir()) == ["ok.md"]


class TestNestedUploadIdentity:
    async def test_nested_paths_land_and_yield_distinct_doc_ids(
        self,
        app: FastAPI,
        settings_env: Callable[..., None],
    ) -> None:
        """The spec_v3 §5.2 identity claim, end to end over the real multipart path.

        Two same-named, same-content files in different subfolders must land
        at their declared relative paths and come out of the real loader as
        two documents with **distinct** ``doc_id``s (structure is identity:
        flattened, the second would silently replace the first).
        """
        from tests.unit.test_loader import FakeBM25, FakeEmbeddings, FakeStore
        from varagity.ingest.loader import ingest_corpus
        from varagity.stores.records import content_hash, derive_doc_id

        docs_root = Path(get_settings().DOCS_PATH)
        content = b"The same quarterly notes, long enough to clear the extraction guard."
        files = [
            ("files", ("notes.md", content, "text/markdown")),
            ("files", ("notes.md", content, "text/markdown")),
        ]
        response = await request(
            app,
            "POST",
            "/api/documents",
            files=files,
            data={"paths": ["q3/notes.md", "q4/notes.md"]},
        )
        assert response.status_code == 201
        entries = response.json()["files"]
        assert [entry["relative_path"] for entry in entries] == ["q3/notes.md", "q4/notes.md"]
        assert (docs_root / "q3" / "notes.md").read_bytes() == content
        assert (docs_root / "q4" / "notes.md").read_bytes() == content

        # The real loader over the uploaded tree (fake stores/embeddings —
        # what's under test is identity derivation, not the databases).
        settings_env(CONTEXTUALIZE="false", EMBEDDING_MODEL="test-model")
        store = FakeStore()
        summary = ingest_corpus(
            str(docs_root),
            store=store,  # type: ignore[arg-type]
            bm25=FakeBM25(),  # type: ignore[arg-type]
            embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
            llm=None,
            verbose=0,
        )
        assert summary.ingested == 2
        file_hash = content_hash(content)
        expected = {
            derive_doc_id("q3/notes.md", file_hash),
            derive_doc_id("q4/notes.md", file_hash),
        }
        assert set(store.documents) == expected
        assert len(expected) == 2  # distinct ids from identical bytes — the path is the identity


class TestDocumentDelete:
    async def test_delete_removes_chunks_from_both_stores(
        self, app: FastAPI, stores: EphemeralStores
    ) -> None:
        seed_document(
            stores, "docKeep", "/docs/keep.txt", ["the kelp corridor stays"], start_index=0
        )
        seed_document(
            stores, "docGone", "/docs/gone.txt", ["petrel platform text", "more petrel"], 10
        )

        listed = (await request(app, "GET", "/api/documents")).json()
        assert {d["doc_id"] for d in listed} == {"docKeep", "docGone"}

        response = await request(app, "DELETE", "/api/documents/docGone")
        assert response.status_code == 200
        assert response.json() == {
            "doc_id": "docGone",
            "chunks_deleted": 2,
            "file_removed": False,
        }

        # pgvector: rows gone (documents marker + chunks via cascade).
        with psycopg.connect(stores.pg_conninfo) as conn:
            docs = conn.execute("SELECT doc_id FROM documents").fetchall()
            chunks = conn.execute("SELECT DISTINCT doc_id FROM chunks").fetchall()
        assert docs == [("docKeep",)]
        assert chunks == [("docKeep",)]

        # Elasticsearch: the deleted doc's chunks no longer match; the
        # other document's still do (the delete was scoped, not a wipe).
        assert stores.bm25.search("petrel", k=10, verbose=0) == []
        assert len(stores.bm25.search("kelp", k=10, verbose=0)) == 1

    async def test_unknown_document_is_a_404(self, app: FastAPI) -> None:
        response = await request(app, "DELETE", "/api/documents/absent")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "document_not_found"
