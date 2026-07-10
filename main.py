"""Varagity entrypoint — delegates to :func:`varagity.cli.app.run`.

Kept thin (spec §5): parsing, dispatch, and rendering live in the
``varagity.cli`` package so they are importable and testable.
"""

from varagity.cli.app import run

if __name__ == "__main__":
    raise SystemExit(run())
