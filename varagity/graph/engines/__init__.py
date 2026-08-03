"""Graph engines — one adapter per file, discovered via registry.

Importing this package imports every adapter module so each
``@register``-decorated engine self-registers (the spec §5.1 pattern applied
to the graph-engine family — spec_graphrag §5.2). Since ADR-017 there is one
seat: ``lightrag``. The losing bake-off adapters (``cognee``, ``graphiti``)
were deleted at stage-2 start — git history preserves them, and **the seam is
what stays**, so the decision remains benchmark-revisitable exactly as the
retriever and chat-engine registries are.

**Importing this package must not import a single engine library.** The
adapter imports ``lightrag`` inside
:meth:`~varagity.graph.base.GraphEngine.session` (stage-1 decision #8) even
though it is a main dependency now, so the registry stays free, unit tests
never touch the engine, and CI's collection cost is unchanged.
``tests/unit/test_graph_engines.py`` fails if that ever regresses.
"""

from varagity.graph.base import (
    GRAPH_ENGINE_REGISTRY,
    GraphEngine,
    GraphSession,
    get_graph_engine,
    register,
)
from varagity.graph.engines import lightrag as _lightrag  # noqa: F401  (self-registration import)

__all__ = [
    "GRAPH_ENGINE_REGISTRY",
    "GraphEngine",
    "GraphSession",
    "get_graph_engine",
    "register",
]
