"""Argument types shared by the command-line entry points.

Each one rejects a bad value at parse time, with the flag's name in the error,
instead of letting it surface later as a misleading failure.
"""

import argparse
import math


def positive_int(value: str) -> int:
    """Parse a command-line integer that must be at least 1.

    Args:
        value: The raw argument.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If the value is below 1.
    """
    number = int(value)
    if number < 1:
        msg = f"must be at least 1, got {number}"
        raise argparse.ArgumentTypeError(msg)
    return number


def positive_float(value: str) -> float:
    """Parse a finite command-line number that must be above 0.

    Args:
        value: The raw argument.

    Returns:
        The parsed number.

    Raises:
        argparse.ArgumentTypeError: If the value is 0 or below, or not finite.
    """
    number = float(value)
    if not (math.isfinite(number) and number > 0):
        msg = f"must be a finite number above 0, got {number}"
        raise argparse.ArgumentTypeError(msg)
    return number


def non_negative_float(value: str) -> float:
    """Parse a finite command-line number that must be 0 or more.

    Args:
        value: The raw argument.

    Returns:
        The parsed number.

    Raises:
        argparse.ArgumentTypeError: If the value is below 0, or not finite.
    """
    number = float(value)
    if not (math.isfinite(number) and number >= 0):
        msg = f"must be a finite number, 0 or more, got {number}"
        raise argparse.ArgumentTypeError(msg)
    return number
