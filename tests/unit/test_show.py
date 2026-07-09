"""Unit tests for varagity.debug.show."""

import pytest

from varagity.debug.show import VERBOSE_LEVELS, check_verbose


def test_supported_levels_are_0_1_2() -> None:
    assert VERBOSE_LEVELS == (0, 1, 2)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_valid_level_passes_through(level: int) -> None:
    assert check_verbose(level) == level


@pytest.mark.parametrize("level", [-1, 3, 42])
def test_out_of_range_level_raises(level: int) -> None:
    with pytest.raises(ValueError, match="verbose must be one of"):
        check_verbose(level)


def test_non_int_level_raises() -> None:
    with pytest.raises(ValueError, match="verbose must be one of"):
        check_verbose("1")  # type: ignore[arg-type]
