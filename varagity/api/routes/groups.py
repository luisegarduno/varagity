"""Conversation-group CRUD routes.

Sidebar groups are user-created folders over the conversation list. The
endpoints mirror :mod:`varagity.api.routes.conversations` — synchronous
thin adapters (FastAPI runs them in its threadpool) over the same
per-request :class:`~varagity.stores.conversation_store.ConversationStore`.
There is no rename: like conversations, a group's identity is its id, and
the GUI offers create/delete only.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from varagity.api.deps import get_conversation_store
from varagity.api.schemas import GroupCreateRequest, GroupOut
from varagity.stores.conversation_store import ConversationStore

router = APIRouter(tags=["groups"])

StoreDep = Annotated[ConversationStore, Depends(get_conversation_store)]


def group_not_found(group_id: str) -> HTTPException:
    """Build the structured 404 for an unknown group.

    Shared with the conversations router, whose move endpoint validates its
    target group against the same vocabulary.

    Args:
        group_id: The id that failed to resolve.

    Returns:
        The exception carrying the ``{code, message}`` detail dict.
    """
    return HTTPException(
        status_code=404,
        detail={
            "code": "group_not_found",
            "message": f"No conversation group with id {group_id!r}",
        },
    )


@router.get("/api/groups")
def list_groups(store: StoreDep) -> list[GroupOut]:
    """List every conversation group, name order.

    Args:
        store: The per-request conversation store.

    Returns:
        All groups, including empty ones.
    """
    return [GroupOut(**group.model_dump()) for group in store.list_groups()]


@router.post("/api/groups", status_code=201)
def create_group(payload: GroupCreateRequest, store: StoreDep) -> GroupOut:
    """Create a conversation group.

    Args:
        payload: The new group's display name.
        store: The per-request conversation store.

    Returns:
        The created group.
    """
    return GroupOut(**store.create_group(payload.name).model_dump())


@router.delete("/api/groups/{group_id}", status_code=204)
def delete_group(group_id: str, store: StoreDep) -> Response:
    """Delete a group; its conversations survive, ungrouped.

    Args:
        group_id: The group to delete.
        store: The per-request conversation store.

    Returns:
        An empty ``204`` response.

    Raises:
        HTTPException: ``404 group_not_found`` for an unknown id.
    """
    if store.delete_group(group_id) == 0:
        raise group_not_found(group_id)
    return Response(status_code=204)
