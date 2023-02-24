#!/usr/bin/env python

"""Utility functions used within the torchfl package."""

from typing import Any


def _get_enum_values(enum_class: Any) -> list[str]:
    """Return a list of values of an enum class."""
    return [x.value for x in enum_class._member_map_.values()]
