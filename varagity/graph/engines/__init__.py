"""Graph engines — one adapter per file, discovered via registry.

Importing this package imports every adapter module so each
``@register``-decorated engine self-registers (the spec §5.1 pattern applied
to the graph-engine family — spec_graphrag §5.2). The three registered here
are the ADR-017 bake-off seats: ``lightrag``, ``cognee``, and ``graphiti``.

**Importing this package must not import a single engine library.** Each
adapter imports its heavy dependency inside
:meth:`~varagity.graph.base.GraphEngine.session` (plan decision #8), so the
registry is free, CI never installs the ``bakeoff`` dependency group, and a
machine with only one engine installed can still run the harness against that
one. ``tests/unit/test_graph_engines.py`` fails if that ever regresses.

The losing adapters are deleted once ADR-017 lands (plan decision #14); the
protocol and this registry are what stay, so the decision remains
benchmark-revisitable.
"""

from varagity.graph.base import (
    GRAPH_ENGINE_REGISTRY,
    GraphEngine,
    GraphSession,
    get_graph_engine,
    register,
)
from varagity.graph.engines import cognee as _cognee  # noqa: F401  (self-registration import)
from varagity.graph.engines import graphiti as _graphiti  # noqa: F401  (self-registration import)
from varagity.graph.engines import lightrag as _lightrag  # noqa: F401  (self-registration import)

__all__ = [
    "GRAPH_ENGINE_REGISTRY",
    "GraphEngine",
    "GraphSession",
    "get_graph_engine",
    "register",
]
