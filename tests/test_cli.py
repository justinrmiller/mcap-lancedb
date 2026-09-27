"""Tests for the argument types the command-line entry points share."""

import argparse
from collections.abc import Callable

import pytest

from mcap_lancedb.cli import non_negative_float, positive_float, positive_int


@pytest.mark.parametrize(
    ("parse", "value"),
    [
        (positive_int, "0"),
        (positive_int, "-3"),
        (positive_float, "0"),
        (positive_float, "nan"),
        (positive_float, "inf"),
        (non_negative_float, "-0.5"),
        (non_negative_float, "nan"),
        (non_negative_float, "inf"),
    ],
)
def test_out_of_range_values_are_rejected(
    parse: Callable[[str], float], value: str
) -> None:
    """Zero, negatives and non-finite numbers fail where each type says so."""
    with pytest.raises(argparse.ArgumentTypeError):
        parse(value)


def test_in_range_values_parse() -> None:
    """Each type returns the parsed number, boundaries included."""
    assert positive_int("1") == 1
    assert positive_float("0.5") == 0.5
    assert non_negative_float("0") == 0.0
