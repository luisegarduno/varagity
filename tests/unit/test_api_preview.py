"""Unit tests for the preview routes (ADR-010): locate + page image.

Same double pattern as ``test_api_documents``: the ``FakeVectorStore``
resolves ``doc_id`` → :class:`DocumentInfo`, while the filesystem side —
containment, content-hash verification, pdfium — runs for real against
fixture PDFs copied under a tmp ``DOCS_PATH``.
"""

import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.unit.test_api_documents import FakeVectorStore, make_app, request
from varagity.stores.records import DocumentInfo, content_hash

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "corpus"

PAGE_2_TEXT = (
    "During each winter season the observatory archives 7.3 terabytes of lidar "
    "returns to vaults on the mainland."
)


def ingested_info(doc_id: str, source: Path) -> DocumentInfo:
    """A DocumentInfo whose content_hash matches the on-disk bytes."""
    return DocumentInfo(
        doc_id=doc_id,
        source=str(source),
        file_type=source.suffix.lstrip("."),
        content_hash=content_hash(source.read_bytes()),
        n_chunks=3,
        ingested_at=datetime(2026, 7, 16, tzinfo=UTC),
        extraction_mix={"text": 3},
    )


@pytest.fixture
def docs_root(tmp_path: Path, settings_env: Callable[..., None]) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    settings_env(DOCS_PATH=str(root))
    return root


@pytest.fixture
def pdf_doc(docs_root: Path) -> DocumentInfo:
    source = docs_root / "saltmere_observatory.pdf"
    shutil.copyfile(FIXTURES / "saltmere_observatory.pdf", source)
    return ingested_info("d0c5a17me4e0bs16", source)


async def locate(app, doc_id: str, text: str = PAGE_2_TEXT):  # type: ignore[no-untyped-def]
    return await request(
        app, "POST", f"/api/documents/{doc_id}/preview/locate", json={"text": text}
    )


class TestLocateRoute:
    async def test_unknown_document_is_a_structured_404(self, docs_root: Path) -> None:
        response = await locate(make_app(vector=FakeVectorStore()), "nope")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "document_not_found"

    async def test_happy_locate_returns_page_rects_and_coverage(
        self, pdf_doc: DocumentInfo
    ) -> None:
        response = await locate(make_app(vector=FakeVectorStore([pdf_doc])), pdf_doc.doc_id)
        assert response.status_code == 200
        body = response.json()
        assert body["available"] is True
        assert body["reason"] is None
        assert body["page"] == 2
        assert body["page_count"] == 2
        assert body["coverage"] > 0.9
        assert body["rects"]
        for rect in body["rects"]:
            assert 0.0 <= rect["x0"] < rect["x1"] <= 1.0
            assert 0.0 <= rect["y0"] < rect["y1"] <= 1.0

    async def test_kill_switch_degrades_to_preview_disabled(
        self, pdf_doc: DocumentInfo, settings_env: Callable[..., None]
    ) -> None:
        settings_env(PREVIEW_ENABLED="false")
        response = await locate(make_app(vector=FakeVectorStore([pdf_doc])), pdf_doc.doc_id)
        assert response.status_code == 200
        assert response.json() == {
            "available": False,
            "reason": "preview_disabled",
            "page": None,
            "page_count": None,
            "rects": [],
            "coverage": None,
        }

    async def test_markdown_source_is_unsupported_type(self, docs_root: Path) -> None:
        source = docs_root / "notes.md"
        source.write_text("# notes")
        info = ingested_info("d1", source)
        response = await locate(make_app(vector=FakeVectorStore([info])), "d1")
        assert response.json() == {
            "available": False,
            "reason": "unsupported_type",
            "page": None,
            "page_count": None,
            "rects": [],
            "coverage": None,
        }

    async def test_deleted_file_degrades_to_file_missing(self, pdf_doc: DocumentInfo) -> None:
        Path(pdf_doc.source).unlink()
        response = await locate(make_app(vector=FakeVectorStore([pdf_doc])), pdf_doc.doc_id)
        assert response.json()["reason"] == "file_missing"

    async def test_source_outside_docs_path_degrades_to_file_missing(
        self, docs_root: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "elsewhere.pdf"
        shutil.copyfile(FIXTURES / "saltmere_observatory.pdf", outside)
        info = ingested_info("d1", outside)
        response = await locate(make_app(vector=FakeVectorStore([info])), "d1")
        assert response.json()["reason"] == "file_missing"

    async def test_edited_file_degrades_to_file_changed(self, pdf_doc: DocumentInfo) -> None:
        """An edit without a reingest must not preview the wrong bytes."""
        Path(pdf_doc.source).write_bytes(b"%PDF-1.4 tampered")
        response = await locate(make_app(vector=FakeVectorStore([pdf_doc])), pdf_doc.doc_id)
        assert response.json()["reason"] == "file_changed"

    async def test_pptx_degrades_to_conversion_unavailable_in_phase_1(
        self, docs_root: Path
    ) -> None:
        source = docs_root / "petrel_turbine_briefing.pptx"
        shutil.copyfile(FIXTURES / "petrel_turbine_briefing.pptx", source)
        info = ingested_info("d1", source)
        response = await locate(make_app(vector=FakeVectorStore([info])), "d1")
        assert response.status_code == 200  # degradable, never a 500
        assert response.json()["reason"] == "conversion_unavailable"

    async def test_corrupt_pdf_degrades_instead_of_500ing(self, docs_root: Path) -> None:
        """Bytes that hash-match but aren't a PDF: pdfium failure is contained."""
        source = docs_root / "broken.pdf"
        source.write_bytes(b"not a pdf at all")
        info = ingested_info("d1", source)  # hash matches the garbage bytes
        response = await locate(make_app(vector=FakeVectorStore([info])), "d1")
        assert response.status_code == 200
        assert response.json()["reason"] == "conversion_failed"

    async def test_no_textual_match_reports_no_match_with_diagnostics(
        self, pdf_doc: DocumentInfo
    ) -> None:
        response = await locate(
            make_app(vector=FakeVectorStore([pdf_doc])),
            pdf_doc.doc_id,
            text="zorblatt quuxification fnord manifold retrograde",
        )
        body = response.json()
        assert body["available"] is False
        assert body["reason"] == "no_match"
        assert body["page"] is None
        assert body["page_count"] == 2
        assert body["coverage"] < 0.3

    async def test_oversized_text_is_a_422(self, pdf_doc: DocumentInfo) -> None:
        response = await locate(
            make_app(vector=FakeVectorStore([pdf_doc])), pdf_doc.doc_id, text="x" * 20_001
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_empty_text_is_a_422(self, pdf_doc: DocumentInfo) -> None:
        response = await locate(make_app(vector=FakeVectorStore([pdf_doc])), pdf_doc.doc_id, "")
        assert response.status_code == 422


class TestPageRoute:
    async def test_serves_an_immutable_png(self, pdf_doc: DocumentInfo) -> None:
        app = make_app(vector=FakeVectorStore([pdf_doc]))
        response = await request(app, "GET", f"/api/documents/{pdf_doc.doc_id}/preview/page/2")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
        # Rendered at PREVIEW_RENDER_WIDTH (default 1536) — IHDR width field.
        assert int.from_bytes(response.content[16:20], "big") == 1536

    async def test_unknown_document_is_a_structured_404(self, docs_root: Path) -> None:
        app = make_app(vector=FakeVectorStore())
        response = await request(app, "GET", "/api/documents/nope/preview/page/1")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "document_not_found"

    async def test_page_out_of_range_is_a_404(self, pdf_doc: DocumentInfo) -> None:
        app = make_app(vector=FakeVectorStore([pdf_doc]))
        response = await request(app, "GET", f"/api/documents/{pdf_doc.doc_id}/preview/page/9")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "page_out_of_range"

    async def test_degrade_reason_is_the_404_code(
        self, pdf_doc: DocumentInfo, settings_env: Callable[..., None]
    ) -> None:
        """An <img> can't read a JSON envelope — the reason rides as the code."""
        settings_env(PREVIEW_ENABLED="false")
        app = make_app(vector=FakeVectorStore([pdf_doc]))
        response = await request(app, "GET", f"/api/documents/{pdf_doc.doc_id}/preview/page/2")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "preview_disabled"

    async def test_edited_file_404s_rather_than_serving_a_lying_image(
        self, pdf_doc: DocumentInfo
    ) -> None:
        Path(pdf_doc.source).write_bytes(b"%PDF-1.4 tampered")
        app = make_app(vector=FakeVectorStore([pdf_doc]))
        response = await request(app, "GET", f"/api/documents/{pdf_doc.doc_id}/preview/page/2")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "file_changed"
