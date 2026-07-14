"""Unit tests for the corpus document routes (upload / list / delete).

Uploads run against a tmp ``DOCS_PATH`` through the real validation and
write path; list/delete fake the two store dependencies (the both-stores
delete against real containers lives in the integration suite).
"""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from elastic_transport import ConnectionError as ESConnectionError
from fastapi import FastAPI

from varagity.api.deps import get_bm25_store, get_vector_store
from varagity.api.main import create_app
from varagity.stores.records import DocumentInfo


class FakeVectorStore:
    """In-memory documents-table double."""

    def __init__(self, documents: list[DocumentInfo] | None = None) -> None:
        self.documents = list(documents or [])
        self.deleted: list[str] = []

    def list_documents(self) -> list[DocumentInfo]:
        return list(self.documents)

    def delete_document(self, doc_id: str) -> int:
        before = len(self.documents)
        self.documents = [info for info in self.documents if info.doc_id != doc_id]
        self.deleted.append(doc_id)
        return before - len(self.documents)


class FakeBM25:
    """delete_document-only double; optionally raises like a downed ES."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.deleted: list[str] = []

    def delete_document(self, doc_id: str) -> int:
        if self.error is not None:
            raise self.error
        self.deleted.append(doc_id)
        return 3


def make_info(doc_id: str, source: str, n_chunks: int = 4) -> DocumentInfo:
    return DocumentInfo(
        doc_id=doc_id,
        source=source,
        file_type=Path(source).suffix.lstrip("."),
        n_chunks=n_chunks,
        ingested_at=datetime(2026, 7, 13, tzinfo=UTC),
        extraction_mix={"text": n_chunks - 1, "ocr_fallback": 1} if n_chunks else {},
    )


@pytest.fixture
def docs_root(tmp_path: Path, settings_env: Callable[..., None]) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    settings_env(DOCS_PATH=str(root), UPLOAD_MAX_MB=1)
    return root


def make_app(vector: FakeVectorStore | None = None, bm25: FakeBM25 | None = None) -> FastAPI:
    application = create_app()
    if vector is not None:
        application.dependency_overrides[get_vector_store] = lambda: vector
    if bm25 is not None:
        application.dependency_overrides[get_bm25_store] = lambda: bm25
    return application


async def request(app: FastAPI, method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        return await client.request(method, path, **kwargs)


def upload_part(name: str, content: bytes) -> tuple[str, tuple[str, bytes, str]]:
    return ("files", (name, content, "application/octet-stream"))


class TestUpload:
    async def test_valid_file_lands_in_docs_path(self, docs_root: Path) -> None:
        response = await request(
            make_app(), "POST", "/api/documents", files=[upload_part("notes.md", b"# hi there")]
        )
        assert response.status_code == 201
        (entry,) = response.json()["files"]
        assert entry == {
            "file_name": "notes.md",
            "size_bytes": 10,
            "stored": True,
            "replaced": False,
            "reason": None,
        }
        assert (docs_root / "notes.md").read_bytes() == b"# hi there"

    async def test_client_path_components_are_stripped(self, docs_root: Path) -> None:
        response = await request(
            make_app(),
            "POST",
            "/api/documents",
            files=[upload_part("../../evil.txt", b"content here")],
        )
        (entry,) = response.json()["files"]
        assert entry["file_name"] == "evil.txt"
        assert (docs_root / "evil.txt").exists()
        assert not (docs_root.parent / "evil.txt").exists()

    async def test_disallowed_extension_is_rejected_per_file(self, docs_root: Path) -> None:
        response = await request(
            make_app(),
            "POST",
            "/api/documents",
            files=[upload_part("good.txt", b"fine"), upload_part("bad.exe", b"nope")],
        )
        assert response.status_code == 201  # mixed batch: the good file landed
        entries = {e["file_name"]: e for e in response.json()["files"]}
        assert entries["good.txt"]["stored"] is True
        assert entries["bad.exe"] == {
            "file_name": "bad.exe",
            "size_bytes": 0,
            "stored": False,
            "replaced": False,
            "reason": "extension_not_allowed",
        }
        assert not (docs_root / "bad.exe").exists()

    async def test_oversized_file_is_rejected_and_no_partial_remains(self, docs_root: Path) -> None:
        too_big = b"x" * (1024 * 1024 + 1)  # UPLOAD_MAX_MB pinned to 1
        response = await request(
            make_app(), "POST", "/api/documents", files=[upload_part("big.txt", too_big)]
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "no_file_stored"
        assert list(docs_root.iterdir()) == []  # the partial was cleaned up

    async def test_all_rejected_batch_is_a_structured_422(self, docs_root: Path) -> None:
        response = await request(
            make_app(), "POST", "/api/documents", files=[upload_part("bad.exe", b"nope")]
        )
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "no_file_stored"
        assert "extension_not_allowed" in error["message"]

    async def test_same_name_reupload_replaces(self, docs_root: Path) -> None:
        await request(make_app(), "POST", "/api/documents", files=[upload_part("a.txt", b"v1")])
        response = await request(
            make_app(), "POST", "/api/documents", files=[upload_part("a.txt", b"v2 longer")]
        )
        (entry,) = response.json()["files"]
        assert entry["replaced"] is True
        assert (docs_root / "a.txt").read_bytes() == b"v2 longer"

    async def test_dotfile_only_name_is_invalid(self, docs_root: Path) -> None:
        response = await request(
            make_app(), "POST", "/api/documents", files=[upload_part(".txt", b"content")]
        )
        assert response.status_code == 422
        assert "invalid_filename" in response.json()["error"]["message"]

    async def test_unwritable_docs_path_is_a_structured_500(self, docs_root: Path) -> None:
        """A ./docs mount the api user can't write → an actionable 500.

        The first real-world failure: an unhandled PermissionError reaches
        the browser as a CORS-less 500 it can only show as "Failed to
        fetch"; the route must contain it into the structured envelope.
        """
        docs_root.chmod(0o555)
        try:
            response = await request(
                make_app(), "POST", "/api/documents", files=[upload_part("ok.txt", b"fine")]
            )
        finally:
            docs_root.chmod(0o755)  # pytest's tmp cleanup needs the write bit back
        assert response.status_code == 500
        error = response.json()["error"]
        assert error["code"] == "docs_path_not_writable"
        assert "writable" in error["message"]

    async def test_write_failure_in_a_mixed_batch_keeps_per_file_outcomes(
        self, docs_root: Path
    ) -> None:
        """A mixed batch on an unwritable directory still escalates.

        Client-side rejections keep their own reasons; the batch becomes
        the server-side 500 only because nothing landed at all.
        """
        docs_root.chmod(0o555)
        try:
            response = await request(
                make_app(),
                "POST",
                "/api/documents",
                files=[upload_part("ok.txt", b"fine"), upload_part("bad.exe", b"nope")],
            )
        finally:
            docs_root.chmod(0o755)
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "docs_path_not_writable"


class TestList:
    async def test_lists_documents_with_derived_file_name(self, docs_root: Path) -> None:
        vector = FakeVectorStore([make_info("d1", str(docs_root / "report.pdf"))])
        response = await request(make_app(vector=vector), "GET", "/api/documents")
        assert response.status_code == 200
        (entry,) = response.json()
        assert entry["doc_id"] == "d1"
        assert entry["file_name"] == "report.pdf"
        assert entry["file_type"] == "pdf"
        assert entry["n_chunks"] == 4
        assert entry["extraction_mix"] == {"text": 3, "ocr_fallback": 1}

    async def test_empty_corpus_is_an_empty_list(self, docs_root: Path) -> None:
        response = await request(make_app(vector=FakeVectorStore()), "GET", "/api/documents")
        assert response.json() == []


class TestDelete:
    async def test_deletes_from_both_stores(self, docs_root: Path) -> None:
        source = docs_root / "gone.txt"
        source.write_text("bye")
        vector = FakeVectorStore([make_info("d1", str(source))])
        bm25 = FakeBM25()
        response = await request(make_app(vector=vector, bm25=bm25), "DELETE", "/api/documents/d1")
        assert response.status_code == 200
        assert response.json() == {"doc_id": "d1", "chunks_deleted": 4, "file_removed": False}
        assert vector.deleted == ["d1"]
        assert bm25.deleted == ["d1"]
        assert source.exists()  # stores-only GC by default

    async def test_remove_file_unlinks_inside_docs_path(self, docs_root: Path) -> None:
        source = docs_root / "gone.txt"
        source.write_text("bye")
        vector = FakeVectorStore([make_info("d1", str(source))])
        response = await request(
            make_app(vector=vector, bm25=FakeBM25()),
            "DELETE",
            "/api/documents/d1",
            params={"remove_file": "true"},
        )
        assert response.json()["file_removed"] is True
        assert not source.exists()

    async def test_remove_file_refuses_outside_docs_path(
        self, docs_root: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "elsewhere.txt"
        outside.write_text("keep me")
        vector = FakeVectorStore([make_info("d1", str(outside))])
        response = await request(
            make_app(vector=vector, bm25=FakeBM25()),
            "DELETE",
            "/api/documents/d1",
            params={"remove_file": "true"},
        )
        assert response.json()["file_removed"] is False
        assert outside.exists()

    async def test_unknown_document_is_a_structured_404(self, docs_root: Path) -> None:
        response = await request(
            make_app(vector=FakeVectorStore(), bm25=FakeBM25()), "DELETE", "/api/documents/nope"
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "document_not_found"

    async def test_es_down_is_a_structured_503_and_pg_untouched(self, docs_root: Path) -> None:
        vector = FakeVectorStore([make_info("d1", str(docs_root / "x.txt"))])
        bm25 = FakeBM25(error=ESConnectionError("refused"))
        response = await request(make_app(vector=vector, bm25=bm25), "DELETE", "/api/documents/d1")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "es_unreachable"
        assert vector.deleted == []  # ES-first ordering: the marker survived
