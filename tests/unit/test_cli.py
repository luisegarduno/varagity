"""Unit tests for the CLI shell (dispatch, exit codes, summary rendering)."""

from collections.abc import Iterator

import pytest

from varagity.cli import app as cli_app
from varagity.config import get_settings
from varagity.debug.show import console
from varagity.ingest.loader import IngestSummary


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_no_subcommand_prints_help_and_settings(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_app.run([]) == 0
    out = capsys.readouterr().out
    assert "usage: varagity" in out
    assert "Loaded settings" in out


def test_settings_secrets_are_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    cli_app.run([])
    out = capsys.readouterr().out
    assert "POSTGRES_PASSWORD" in out
    assert "change-me" not in out
    assert "***" in out


def test_ingest_dispatch_and_exit_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_ingest(verbose: int) -> IngestSummary:
        calls.append(verbose)
        return IngestSummary(discovered=2, ingested=2, chunks=7)

    monkeypatch.setattr(cli_app, "ingest_corpus", lambda verbose: fake_ingest(verbose))
    with console.capture() as capture:
        exit_code = cli_app.run(["-v", "0", "ingest"])
    assert exit_code == 0
    assert calls == [0]
    out = capture.get()
    assert "Ingest summary" in out
    assert "7" in out


def test_ingest_exit_one_on_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_app, "ingest_corpus", lambda verbose: IngestSummary(discovered=1, failed=1)
    )
    with console.capture():
        assert cli_app.run(["ingest"]) == 1


def test_verbose_flag_overrides_settings_default(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    monkeypatch.setattr(
        cli_app,
        "ingest_corpus",
        lambda verbose: seen.append(verbose) or IngestSummary(),
    )
    with console.capture():
        cli_app.run(["-v", "2", "ingest"])
    assert seen == [2]


def test_verbose_flag_accepted_after_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plan's canonical invocation is `main.py ingest -v 1`."""
    seen: list[int] = []
    monkeypatch.setattr(
        cli_app,
        "ingest_corpus",
        lambda verbose: seen.append(verbose) or IngestSummary(),
    )
    with console.capture():
        cli_app.run(["ingest", "-v", "0"])
        cli_app.run(["-v", "2", "ingest"])  # pre-subcommand value survives
    assert seen == [0, 2]
