"""Conversation CRUD routes (spec_v2 §4.2, §4.4).

Thin adapters over :class:`~varagity.stores.conversation_store
.ConversationStore` — the endpoints are synchronous on purpose (FastAPI
runs them in its threadpool), matching the sync psycopg store underneath.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from varagity.api.deps import get_conversation_store
from varagity.api.schemas import (
    ConversationCreateRequest,
    ConversationDetailOut,
    ConversationSummaryOut,
)
from varagity.stores.conversation_store import ConversationStore

router = APIRouter(tags=["conversations"])

StoreDep = Annotated[ConversationStore, Depends(get_conversation_store)]


def _not_found(conversation_id: str) -> HTTPException:
    """Build the structured 404 for an unknown conversation.

    Args:
        conversation_id: The id that failed to resolve.

    Returns:
        The exception carrying the ``{code, message}`` detail dict.
    """
    return HTTPException(
        status_code=404,
        detail={
            "code": "conversation_not_found",
            "message": f"No conversation with id {conversation_id!r}",
        },
    )


@router.get("/api/conversations")
def list_conversations(store: StoreDep) -> list[ConversationSummaryOut]:
    """List every conversation, most recently updated first.

    Args:
        store: The per-request conversation store.

    Returns:
        Summaries with message counts.
    """
    return [
        ConversationSummaryOut(**summary.model_dump()) for summary in store.list_conversations()
    ]


@router.post("/api/conversations", status_code=201)
def create_conversation(
    payload: ConversationCreateRequest, store: StoreDep
) -> ConversationSummaryOut:
    """Start a conversation.

    Args:
        payload: Optional explicit title (the first chat turn auto-titles
            otherwise).
        store: The per-request conversation store.

    Returns:
        The created conversation's summary.
    """
    return ConversationSummaryOut(**store.create_conversation(payload.title).model_dump())


@router.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str, store: StoreDep) -> ConversationDetailOut:
    """Fetch a full transcript: messages plus each answer's stored sources.

    Args:
        conversation_id: The conversation to fetch.
        store: The per-request conversation store.

    Returns:
        The transcript, oldest message first.

    Raises:
        HTTPException: ``404 conversation_not_found`` for an unknown id.
    """
    detail = store.get_conversation(conversation_id)
    if detail is None:
        raise _not_found(conversation_id)
    return ConversationDetailOut(**detail.model_dump())


@router.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, store: StoreDep) -> Response:
    """Delete a conversation; messages and sources cascade.

    Args:
        conversation_id: The conversation to delete.
        store: The per-request conversation store.

    Returns:
        An empty ``204`` response.

    Raises:
        HTTPException: ``404 conversation_not_found`` for an unknown id.
    """
    if store.delete_conversation(conversation_id) == 0:
        raise _not_found(conversation_id)
    return Response(status_code=204)
