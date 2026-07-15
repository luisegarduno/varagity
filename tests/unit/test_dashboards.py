"""Static guards on the provisioned Grafana dashboards (spec_v3 §6.4).

Every panel expression is parsed and checked against the metric catalog
(:func:`varagity.observability.metrics.catalog`) — no live Prometheus, so
this runs in the default unit suite.

Three classes of bug, one test each:

1. a renamed/removed metric leaving a panel permanently empty;
2. a ``sum by (…)`` on a label the metric never declares (same symptom);
3. ``increase()``/``rate()`` over an ingest counter — the bug v3 actually
   had (spec_v3 §6.1). Ingestion is bursty and its counters are *born at
   their full value* after an API restart (the labelled children don't
   exist until the first increment), so Prometheus never observes the
   ``0 → N`` rise and the delta is ``0`` — over **any** window, which is
   why this rule is window-independent rather than only banning
   ``$__rate_interval`` as the spec proposed. Verified live: with the
   counter at 2, ``increase(varagity_ingest_chunks_total[6h])`` is ``0``,
   and using it as a divisor rendered ``+Inf``.

Unlabelled histograms are *not* subject to (3): they are initialised at 0
on definition, so the rise is observed and ``rate()`` over a wide window is
correct — that asymmetry is exactly why panels 14 and 15 differ.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from varagity.observability.metrics import catalog

DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "observability" / "grafana" / "dashboards"

# prometheus_client strips `_total` from a Counter's internal name, and the
# client appends the histogram suffixes at exposition time; a dashboard
# references the *exposed* names, so both directions are reconciled here.
_SUFFIXES = ("_bucket", "_sum", "_count", "_total")

# Declared per-metric by prometheus_client itself, not by our catalog.
_HISTOGRAM_LABELS = {"le"}

# The counters whose series are born at full value (see the module docstring).
_BURSTY_COUNTER = re.compile(r"(?:increase|rate)\s*\(\s*(varagity_ingest_\w+)")

_METRIC_REF = re.compile(r"varagity_[a-z0-9_]+")
_SELECTOR = re.compile(r"(varagity_[a-z0-9_]+)\s*\{([^}]*)\}")
_BY_CLAUSE = re.compile(r"\bby\s*\(([^)]*)\)")
_LABEL_IN_SELECTOR = re.compile(r"(\w+)\s*(?:=~|!~|!=|=)")


def _dashboards() -> list[Path]:
    """Every provisioned dashboard file.

    Returns:
        The dashboard JSON paths, sorted.
    """
    return sorted(DASHBOARD_DIR.glob("*.json"))


def _panel_exprs(path: Path) -> list[tuple[str, str]]:
    """Extract every panel expression from one dashboard.

    Args:
        path: The dashboard JSON file.

    Returns:
        ``(panel label, expr)`` pairs, panel label being ``id · title``.
    """
    dashboard: dict[str, Any] = json.loads(path.read_text())
    found: list[tuple[str, str]] = []
    for panel in dashboard.get("panels", []):
        label = f"{path.name} panel {panel.get('id')} · {panel.get('title')}"
        for target in panel.get("targets", []):
            expr = target.get("expr")
            if expr:
                found.append((label, expr))
    return found


def _all_exprs() -> list[tuple[str, str]]:
    """Every panel expression across every dashboard.

    Returns:
        ``(panel label, expr)`` pairs.
    """
    return [pair for path in _dashboards() for pair in _panel_exprs(path)]


def _resolve(reference: str, declared: dict[str, tuple[str, ...]]) -> str | None:
    """Map an exposed metric name back to its catalog entry.

    Args:
        reference: The name as a dashboard references it (possibly with a
            ``_bucket``/``_sum``/``_count``/``_total`` suffix).
        declared: The catalog.

    Returns:
        The catalog key, or ``None`` if the metric is not in the catalog.
    """
    if reference in declared:
        return reference
    for suffix in _SUFFIXES:
        if reference.endswith(suffix) and reference[: -len(suffix)] in declared:
            return reference[: -len(suffix)]
    return None


def test_dashboard_dir_has_panels() -> None:
    """The guards below are vacuous if the parser finds nothing."""
    exprs = _all_exprs()
    assert len(_dashboards()) >= 3, "expected the query/ingestion/infra dashboards"
    assert len(exprs) >= 15, f"expected the full panel set, parsed {len(exprs)}"


@pytest.mark.parametrize("label, expr", _all_exprs())
def test_referenced_metrics_exist(label: str, expr: str) -> None:
    """Every ``varagity_*`` metric a panel queries is in the catalog.

    Args:
        label: The panel this expression belongs to.
        expr: The PromQL expression.
    """
    declared = catalog()
    for reference in _METRIC_REF.findall(expr):
        assert _resolve(reference, declared) is not None, (
            f"{label}: queries {reference!r}, which no metric in "
            f"varagity/observability/metrics.py exposes"
        )


@pytest.mark.parametrize("label, expr", _all_exprs())
def test_referenced_labels_are_declared(label: str, expr: str) -> None:
    """Panels only group/filter on labels their metrics declare.

    Args:
        label: The panel this expression belongs to.
        expr: The PromQL expression.
    """
    declared = catalog()
    referenced = [
        name for name in _METRIC_REF.findall(expr) if _resolve(name, declared) is not None
    ]
    if not referenced:
        return  # a non-varagity panel (prefect-exporter, dcgm, `up`)
    available = _HISTOGRAM_LABELS.union(
        *(set(declared[key]) for name in referenced if (key := _resolve(name, declared)))
    )
    for group in _BY_CLAUSE.findall(expr):
        for name in (part.strip() for part in group.split(",")):
            assert name in available, (
                f"{label}: groups by {name!r}, not declared on {sorted(set(referenced))}"
            )
    for metric, selector in _SELECTOR.findall(expr):
        key = _resolve(metric, declared)
        assert key is not None  # filtered above
        for name in _LABEL_IN_SELECTOR.findall(selector):
            assert name in set(declared[key]) | _HISTOGRAM_LABELS, (
                f"{label}: filters {metric} on {name!r}, which it does not declare"
            )


@pytest.mark.parametrize("label, expr", _all_exprs())
def test_no_rate_over_bursty_ingest_counters(label: str, expr: str) -> None:
    """No panel takes a delta over an ingest counter.

    Args:
        label: The panel this expression belongs to.
        expr: The PromQL expression.
    """
    offenders = _BURSTY_COUNTER.findall(expr)
    assert not offenders, (
        f"{label}: increase()/rate() over {offenders} — ingest counters are bursty and are "
        f"born at their full value after a restart, so the delta is 0 over any window "
        f"(spec_v3 §6.1). Use the store-derived varagity_corpus_* gauges for corpus size, "
        f"or a suffix-free cumulative ratio."
    )


def test_ingestion_size_panels_use_corpus_gauges() -> None:
    """The panels that read 0 for all of v2 now read the store.

    Pins the §6.1b intent so a future edit can't quietly reintroduce a
    process-history counter as the answer to "how big is my corpus".
    """
    exprs = dict(_panel_exprs(DASHBOARD_DIR / "ingestion.json"))
    size_panels = [
        (label, expr)
        for label, expr in exprs.items()
        if "corpus" in label.lower() and "rate" not in label.lower()
    ]
    assert size_panels, "no corpus-size panels found on the Ingestion dashboard"
    for label, expr in size_panels:
        assert "varagity_corpus_" in expr, f"{label}: should read a varagity_corpus_* gauge"
