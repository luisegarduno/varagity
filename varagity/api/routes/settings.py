"""``GET``/``PATCH /api/settings`` — the live settings surface (spec_v2 §4.7).

``GET`` returns the full overridable catalog (effective values, groups,
override/reingest flags, registry-derived choices) plus the corpus-stale
flag; ``PATCH`` merges new overrides over the active set, validates the
whole through the config validators, persists to ``app_settings``, clears
the settings cache (via :mod:`varagity.api.runtime_settings`), and flags
the corpus stale when a reingest-affecting knob actually changed on a
non-empty corpus. Query-time knobs take effect on the next question;
ingest-time knobs surface the v1 "hashes don't change" footgun as the
"Re-ingest to apply" affordance instead of a silent no-op.
"""

import logging
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from varagity.api.deps import get_app_settings_store, get_vector_store
from varagity.api.runtime_settings import (
    OVERRIDABLE,
    REINGEST_AFFECTING,
    active_overrides,
    apply_overrides,
    to_env_value,
)
from varagity.api.schemas import SettingOut, SettingsPatchRequest, SettingsResponse
from varagity.config import Settings, get_settings
from varagity.stores.app_settings_store import AppSettingsStore
from varagity.stores.vector_store import ContextualVectorDB

logger = logging.getLogger(__name__)

router = APIRouter(tags=["settings"])

SettingsStoreDep = Annotated[AppSettingsStore, Depends(get_app_settings_store)]
VectorStoreDep = Annotated[ContextualVectorDB, Depends(get_vector_store)]


def _catalog(settings: Settings, overridden: set[str], corpus_stale: bool) -> SettingsResponse:
    """Render the overridable catalog in its stable (grouped) order.

    Args:
        settings: The effective settings.
        overridden: Names with an active runtime override.
        corpus_stale: The persisted stale flag.

    Returns:
        The full response body (shared by ``GET`` and ``PATCH``).
    """
    return SettingsResponse(
        settings=[
            SettingOut(
                name=name,
                value=getattr(settings, name),
                group=spec.group,
                overridden=name in overridden,
                reingest_affecting=spec.reingest_affecting,
                choices=spec.choices() if spec.choices is not None else None,
            )
            for name, spec in OVERRIDABLE.items()
        ],
        corpus_stale=corpus_stale,
    )


@router.get("/api/settings")
def read_settings(store: SettingsStoreDep) -> SettingsResponse:
    """Report the effective settings and the corpus-stale flag.

    Args:
        store: The per-request app-settings store.

    Returns:
        The overridable catalog with effective (env + override) values.
    """
    return _catalog(get_settings(), set(active_overrides()), store.is_corpus_stale())


@router.patch("/api/settings")
def patch_settings(
    payload: SettingsPatchRequest, store: SettingsStoreDep, vector_store: VectorStoreDep
) -> SettingsResponse:
    """Apply, validate, and persist runtime setting overrides.

    A ``None`` value clears that override (reverting to the env value).
    Validation runs on the *merged whole* through every config validator,
    so linked constraints (the fusion weight pair, the rerank bounds) hold
    exactly as they do for ``.env`` — an invalid patch changes nothing.

    Args:
        payload: The overrides to set or clear.
        store: The per-request app-settings store (persistence + flag).
        vector_store: The vector store (the corpus-emptiness check).

    Returns:
        The full post-patch catalog; ``corpus_stale`` reports whether a
        reingest is now needed.

    Raises:
        HTTPException: ``422 unknown_setting`` for a name outside the
            catalog; ``422 invalid_settings`` when the merged settings fail
            validation; ``503 postgres_unreachable`` if persistence fails
            (the overrides are rolled back — the process never diverges
            from the table).
    """
    unknown = sorted(set(payload.overrides) - set(OVERRIDABLE))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unknown_setting",
                "message": f"not overridable: {unknown}; overridable: {sorted(OVERRIDABLE)}",
            },
        )

    before = get_settings()
    before_reingest = {name: getattr(before, name) for name in REINGEST_AFFECTING}

    previous = active_overrides()
    candidate = dict(previous)
    for name, value in payload.overrides.items():
        if value is None:
            candidate.pop(name, None)
        else:
            candidate[name] = to_env_value(value)

    try:
        settings = apply_overrides(candidate)
    except ValidationError as error:
        message = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'settings'}: {item['msg']}"
            for item in error.errors()
        )
        raise HTTPException(
            status_code=422, detail={"code": "invalid_settings", "message": message}
        ) from error

    try:
        for name, value in payload.overrides.items():
            if value is None:
                store.delete_override(name)
            else:
                store.set_override(name, to_env_value(value))
        changed_reingest = sorted(
            name for name in REINGEST_AFFECTING if getattr(settings, name) != before_reingest[name]
        )
        corpus_stale = store.is_corpus_stale()
        if changed_reingest and not corpus_stale and vector_store.document_count() > 0:
            store.set_corpus_stale(True)
            corpus_stale = True
            logger.info(
                "reingest-affecting setting(s) changed (%s) — corpus flagged stale",
                ", ".join(changed_reingest),
            )
    except psycopg.Error as error:
        # The table couldn't record what the process now runs with — roll the
        # process back so a restart can't silently lose the change.
        apply_overrides(previous)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "postgres_unreachable",
                "message": f"could not persist overrides — nothing changed ({error})",
            },
        ) from error

    return _catalog(settings, set(candidate), corpus_stale)
