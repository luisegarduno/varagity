"""Runtime settings override layer (spec_v2 §4.7, plan decision #9).

The GUI tunes the pipeline without editing ``.env``: overrides persist in
the ``app_settings`` table (migration ``002``, via
:class:`~varagity.stores.app_settings_store.AppSettingsStore`) and are
**merged over the env-loaded** :class:`~varagity.config.Settings` with the
exact mechanism the eval harness proved
(``pinned_eval_settings``): export the values as environment variables —
which pydantic-settings gives precedence over ``.env`` — then
``get_settings.cache_clear()``. Every module that reads ``get_settings()``
(the convention) sees the override on its next call, so query-time knobs
take effect on the next question with zero pipeline edits.

This module owns the *process-env* side: the :data:`OVERRIDABLE` catalog
(which settings the GUI may override, grouped per spec §4.7, flagged when
reingest-affecting), atomic apply-with-validation (an invalid merge rolls
the environment back and raises, so a bad ``PATCH`` can never leave the
process half-configured), and the startup replay of persisted rows.
Persistence and the corpus-stale flag live in the store; the route
(:mod:`varagity.api.routes.settings`) orchestrates both.

Single-worker caveat (plan decision #11): the environment is process-global
state, which is exactly why one uvicorn worker serves the API. A PATCH
racing an in-flight request can hand that request the old or new values —
acceptable for the single-user posture, guarded by :data:`_lock` against
torn writes.
"""

import logging
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pydantic import ValidationError

from varagity.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OverridableSetting:
    """Catalog entry for one GUI-overridable setting.

    Attributes:
        group: The spec §4.7 drawer group (``retrieval`` | ``generation`` |
            ``ingestion``).
        reingest_affecting: Whether changing it does **not** change content
            hashes (the v1 footgun) — the corpus goes stale until
            ``ingest --reingest``.
        choices: Lazy provider of the valid values for enum-like settings
            (lazy so registries resolve at request time, reflecting any
            newly registered implementation); ``None`` for free-form and
            numeric settings.
    """

    group: str
    reingest_affecting: bool = False
    choices: Callable[[], list[str]] | None = None


def _retriever_names() -> list[str]:
    """List the registered retrieval methods (lazy registry import).

    Returns:
        Sorted registry names.
    """
    from varagity.retrieval.base import RETRIEVER_REGISTRY

    return sorted(RETRIEVER_REGISTRY)


def _chunker_names() -> list[str]:
    """List the registered chunking strategies (lazy registry import).

    Returns:
        Sorted registry names.
    """
    from varagity.chunking import CHUNKER_REGISTRY

    return sorted(CHUNKER_REGISTRY)


def _ocr_engine_names() -> list[str]:
    """List the available OCR engines (lazy factory import).

    Returns:
        Sorted engine names.
    """
    from varagity.ingest.parsers.pdf import OCR_ENGINE_FACTORIES

    return sorted(OCR_ENGINE_FACTORIES)


def _llm_model_types() -> list[str]:
    """List the chat-capable model-registry aliases.

    Returns:
        The ``CHAT_MODEL_TYPE`` vocabulary.
    """
    from varagity.models.registry import LLM_MODEL_TYPES

    return list(LLM_MODEL_TYPES)


# The GUI-overridable settings (spec §4.7 groups; the Display group is
# client-side — theme and panel preferences never reach the pipeline).
# RERANK_BASE_METHOD's choices mirror config.py's validator ("reranked"
# would recurse), not the retriever registry.
OVERRIDABLE: dict[str, OverridableSetting] = {
    "RETRIEVAL_METHOD": OverridableSetting("retrieval", choices=_retriever_names),
    "TOP_K": OverridableSetting("retrieval"),
    "SEMANTIC_WEIGHT": OverridableSetting("retrieval"),
    "BM25_WEIGHT": OverridableSetting("retrieval"),
    "RERANK_ENABLED": OverridableSetting("retrieval"),
    "RERANK_TOP_N": OverridableSetting("retrieval"),
    "RERANK_BASE_METHOD": OverridableSetting(
        "retrieval", choices=lambda: ["semantic", "bm25", "hybrid"]
    ),
    "RERANK_CANDIDATES": OverridableSetting("retrieval"),
    "LLM_TEMPERATURE": OverridableSetting("generation"),
    "MAX_TOKENS": OverridableSetting("generation"),
    "CHAT_MODEL_TYPE": OverridableSetting("generation", choices=_llm_model_types),
    "CHUNKING_STRATEGY": OverridableSetting(
        "ingestion", reingest_affecting=True, choices=_chunker_names
    ),
    "CHUNK_SIZE": OverridableSetting("ingestion", reingest_affecting=True),
    "CHUNK_OVERLAP": OverridableSetting("ingestion", reingest_affecting=True),
    "CONTEXTUALIZE": OverridableSetting("ingestion", reingest_affecting=True),
    "OCR_ENGINE": OverridableSetting(
        "ingestion", reingest_affecting=True, choices=_ocr_engine_names
    ),
    "ALLOWED_EXTENSIONS": OverridableSetting("ingestion"),
}

REINGEST_AFFECTING: frozenset[str] = frozenset(
    name for name, spec in OVERRIDABLE.items() if spec.reingest_affecting
)
"""The settings whose change marks the corpus stale (spec §4.7)."""

_lock = threading.Lock()
# The overrides currently exported to the process environment.
_active: dict[str, str] = {}
# Pre-override process-env value per key, captured the first time a key is
# overridden so clearing the override restores the original environment
# (including "unset", stored as None).
_baseline_env: dict[str, str | None] = {}


def to_env_value(value: bool | int | float | str) -> str:
    """Convert a JSON scalar to the env-string form pydantic-settings parses.

    Args:
        value: The override value as it arrived in the PATCH body.

    Returns:
        The environment-variable string (bools lowercase, per the ``.env``
        convention).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def active_overrides() -> dict[str, str]:
    """Snapshot the overrides currently applied to the process environment.

    Returns:
        Setting name → env-string value (a copy).
    """
    with _lock:
        return dict(_active)


def _swap_env(overrides: Mapping[str, str]) -> None:
    """Make the process environment reflect exactly ``overrides``.

    Overridable keys present in ``overrides`` are exported (their original
    env value snapshotted once); keys absent are restored to that snapshot.
    The settings cache is cleared so the next ``get_settings()`` rebuilds.

    Caller must hold :data:`_lock`.

    Args:
        overrides: The complete desired override set (env-string values).
    """
    for name in OVERRIDABLE:
        value = overrides.get(name)
        if value is not None:
            if name not in _baseline_env:
                _baseline_env[name] = os.environ.get(name)
            os.environ[name] = value
        elif name in _baseline_env:
            original = _baseline_env.pop(name)
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original
    get_settings.cache_clear()


def apply_overrides(overrides: Mapping[str, str]) -> Settings:
    """Atomically make ``overrides`` the complete active override set.

    The merged environment is validated by constructing
    :class:`~varagity.config.Settings` (every config validator runs — the
    weight pair, the rerank constraints, the vocabularies); a failure rolls
    the environment back to the previous override set and re-raises, so the
    process never serves half-applied settings.

    Args:
        overrides: Setting name → env-string value; must be the *entire*
            desired set (the route merges patches over
            :func:`active_overrides`). Keys must be in :data:`OVERRIDABLE`.

    Returns:
        The now-effective settings.

    Raises:
        KeyError: If a key is not overridable.
        pydantic.ValidationError: If the merged settings are invalid (the
            previous overrides remain in effect).
    """
    unknown = sorted(set(overrides) - set(OVERRIDABLE))
    if unknown:
        raise KeyError(f"not overridable: {unknown}; overridable: {sorted(OVERRIDABLE)}")
    global _active
    with _lock:
        previous = dict(_active)
        _swap_env(overrides)
        try:
            settings = get_settings()
        except ValidationError:
            _swap_env(previous)
            raise
        _active = dict(overrides)
        return settings


def load_persisted_overrides(load: Callable[[], dict[str, str]]) -> None:
    """Replay persisted overrides into the environment (API startup).

    Runs after the migration runner in the lifespan, so overrides survive an
    ``api`` restart. Rows that no longer validate (e.g. a chunker override
    from a removed registry entry) or are no longer overridable are dropped
    from the *applied* set with an error log — the API must boot on env
    defaults rather than crash on a stale row; the row itself stays put for
    inspection.

    Args:
        load: Zero-argument provider of the persisted rows (a bound
            ``AppSettingsStore.load_overrides``; injected so unit tests
            need no database).
    """
    persisted = load()
    known = {name: value for name, value in persisted.items() if name in OVERRIDABLE}
    dropped = sorted(set(persisted) - set(known))
    if dropped:
        logger.error("ignoring persisted overrides that are no longer overridable: %s", dropped)
    if not known:
        return
    try:
        apply_overrides(known)
    except ValidationError as error:
        logger.error(
            "persisted overrides no longer validate — starting on env defaults "
            "(fix or clear them via PATCH /api/settings): %s",
            error,
        )
        return
    logger.info("applied %d persisted setting override(s): %s", len(known), sorted(known))


def reset_for_tests() -> None:
    """Restore the un-overridden environment (test isolation helper).

    Clears every active override, restores the baseline env snapshot, and
    clears the settings cache — the moral equivalent of the ``settings_env``
    fixture's teardown for this module's state.
    """
    with _lock:
        _swap_env({})
        _active.clear()
