"""Unit tests for golden dataset loading, validation, and resolution."""

from collections.abc import Callable
from pathlib import Path

import pytest

from varagity.eval.datasets import GoldenEntry, load_golden, resolve_golden
from varagity.eval.evaluate import EVAL_CORPUS, GOLDEN_PATH, PINNED_EVAL_SETTINGS
from varagity.stores.records import content_hash, derive_doc_id

VALID_LINE = '{"query": "who?", "relevant": [{"rel_source": "a.md", "chunk_index": 0}]}'


def _write_golden(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestLoadGolden:
    def test_loads_entries_in_order(self, tmp_path: Path) -> None:
        path = _write_golden(
            tmp_path,
            '{"query": "q1", "relevant": [{"rel_source": "a.md", "chunk_index": 0}]}',
            '{"query": "q2", "relevant": [{"rel_source": "a.md", "chunk_index": 1}, '
            '{"rel_source": "b.txt", "chunk_index": 2}]}',
        )
        entries = load_golden(path)
        assert [entry.query for entry in entries] == ["q1", "q2"]
        assert entries[1].relevant[1].rel_source == "b.txt"
        assert entries[1].relevant[1].chunk_index == 2

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = _write_golden(tmp_path, VALID_LINE, "", "   ", VALID_LINE)
        assert len(load_golden(path)) == 2

    def test_invalid_json_names_the_line(self, tmp_path: Path) -> None:
        path = _write_golden(tmp_path, VALID_LINE, "{not json")
        with pytest.raises(ValueError, match=r":2: not valid JSON"):
            load_golden(path)

    @pytest.mark.parametrize(
        "bad_line",
        [
            '{"relevant": [{"rel_source": "a.md", "chunk_index": 0}]}',  # no query
            '{"query": "", "relevant": [{"rel_source": "a.md", "chunk_index": 0}]}',
            '{"query": "q", "relevant": []}',  # nothing relevant
            '{"query": "q", "relevant": [{"rel_source": "", "chunk_index": 0}]}',
            '{"query": "q", "relevant": [{"rel_source": "a.md", "chunk_index": -1}]}',
        ],
    )
    def test_schema_violations_name_the_line(self, tmp_path: Path, bad_line: str) -> None:
        path = _write_golden(tmp_path, VALID_LINE, bad_line)
        with pytest.raises(ValueError, match=r":2: invalid golden entry"):
            load_golden(path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = _write_golden(tmp_path, "")
        with pytest.raises(ValueError, match="empty"):
            load_golden(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_golden(tmp_path / "nope.jsonl")


class TestResolveGolden:
    def test_chunk_ids_derive_from_relative_path_and_bytes(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_bytes(b"alpha doc")
        (tmp_path / "b.txt").write_bytes(b"beta doc")
        entries = [
            GoldenEntry.model_validate(
                {
                    "query": "q",
                    "relevant": [
                        {"rel_source": "a.md", "chunk_index": 1},
                        {"rel_source": "b.txt", "chunk_index": 0},
                    ],
                }
            )
        ]

        resolved = resolve_golden(entries, tmp_path)

        doc_a = derive_doc_id("a.md", content_hash(b"alpha doc"))
        doc_b = derive_doc_id("b.txt", content_hash(b"beta doc"))
        assert resolved[0].chunk_ids == [f"{doc_a}::1", f"{doc_b}::0"]
        # The portable refs ride along for reporting.
        assert resolved[0].relevant == entries[0].relevant

    def test_same_source_resolves_to_one_doc_id(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_bytes(b"alpha doc")
        entries = [
            GoldenEntry.model_validate(
                {
                    "query": "q",
                    "relevant": [
                        {"rel_source": "a.md", "chunk_index": 0},
                        {"rel_source": "a.md", "chunk_index": 2},
                    ],
                }
            )
        ]
        chunk_ids = resolve_golden(entries, tmp_path)[0].chunk_ids
        assert chunk_ids[0].split("::")[0] == chunk_ids[1].split("::")[0]

    def test_missing_corpus_file_raises(self, tmp_path: Path) -> None:
        entries = [
            GoldenEntry.model_validate(
                {"query": "q", "relevant": [{"rel_source": "ghost.md", "chunk_index": 0}]}
            )
        ]
        with pytest.raises(FileNotFoundError, match="ghost.md"):
            resolve_golden(entries, tmp_path)


class TestShippedGoldenDataset:
    """The checked-in golden set stays in sync with the fixtures corpus."""

    def test_loads_and_resolves_against_the_fixtures_corpus(self) -> None:
        entries = load_golden(GOLDEN_PATH)
        assert len(entries) >= 15  # the plan's ~15–20 hand-authored entries
        resolved = resolve_golden(entries, EVAL_CORPUS)  # every rel_source exists
        assert len(resolved) == len(entries)

    def test_text_refs_are_within_actual_chunk_counts(
        self, settings_env: Callable[..., None]
    ) -> None:
        """Golden chunk_index values match the pinned chunker's boundaries.

        Covers the .txt/.md refs only — PDF parsing (Docling/OCR) is far too
        slow for the unit suite; PDF refs are validated against the store by
        the harness at eval time.
        """
        settings_env(**PINNED_EVAL_SETTINGS)
        from varagity.chunking import get_chunker
        from varagity.ingest.parsers import get_parser

        parser = get_parser("text")
        chunker = get_chunker("recursive_character")
        counts: dict[str, int] = {}
        for entry in load_golden(GOLDEN_PATH):
            for ref in entry.relevant:
                if not ref.rel_source.endswith((".txt", ".md")):
                    continue
                if ref.rel_source not in counts:
                    raw = parser.extract(EVAL_CORPUS / ref.rel_source, verbose=0)
                    counts[ref.rel_source] = len(
                        chunker.split(raw.text, source_meta=raw.source_meta, verbose=0)
                    )
                assert ref.chunk_index < counts[ref.rel_source], (
                    f"{ref.rel_source} chunk_index {ref.chunk_index} is out of range "
                    f"({counts[ref.rel_source]} chunks) — regenerate the golden set"
                )

    def test_scanned_queries_exist_for_the_ocr_benchmark(self) -> None:
        from varagity.eval.ocr_benchmark import OCR_FIXTURE_PAGES, _scanned_entries

        resolved = resolve_golden(load_golden(GOLDEN_PATH), EVAL_CORPUS)
        scanned = _scanned_entries(resolved)
        assert len(scanned) >= 2, "the OCR benchmark needs scanned-doc golden queries"
        for name in OCR_FIXTURE_PAGES:
            assert (EVAL_CORPUS / name).is_file()

    def test_ocr_ground_truth_files_exist_for_every_fixture(self) -> None:
        from varagity.eval.ocr_benchmark import OCR_FIXTURE_PAGES, OCR_TRUTH_DIR

        for name in OCR_FIXTURE_PAGES:
            truth = (OCR_TRUTH_DIR / name).with_suffix(".txt")
            assert truth.is_file(), f"missing OCR ground truth {truth}"
            assert truth.read_text(encoding="utf-8").strip()
