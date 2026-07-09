"""Rich-rendered console helpers backing the ``verbose=`` parameter convention.

Varagity separates three output channels (spec §14); this module implements
the first — *human-facing console output* — with three levels:

* ``0`` — off: render nothing.
* ``1`` — low: names and counts.
* ``2`` — high: full metadata, rich panels.

Conventions:

* Every public function in the codebase accepts
  ``verbose: int = settings.DEFAULT_VERBOSE`` and raises :class:`ValueError`
  on invalid levels (enforced via :func:`check_verbose`).
* All rendering lives here as ``v_<function_name>(...)`` helpers (e.g.
  ``v_discover``, ``v_chunk``, ``v_retrieve``), keeping presentation out of
  business logic. Helpers render nothing at level ``0``.

Concrete ``v_<name>`` helpers land alongside the features they render.
"""

VERBOSE_LEVELS: tuple[int, ...] = (0, 1, 2)


def check_verbose(verbose: int) -> int:
    """Validate a ``verbose`` level.

    Called at the top of every function that accepts a ``verbose`` parameter,
    so an invalid level fails fast instead of silently rendering nothing.

    Args:
        verbose: Requested verbosity; must be 0 (off), 1 (low), or 2 (high).

    Returns:
        The validated level, unchanged.

    Raises:
        ValueError: If ``verbose`` is not one of :data:`VERBOSE_LEVELS`.
    """
    if verbose not in VERBOSE_LEVELS:
        raise ValueError(
            f"verbose must be one of {VERBOSE_LEVELS} (0=off, 1=low, 2=high); got {verbose!r}"
        )
    return verbose
