#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Defines the constants to ensure the consistency and compatibility between the files."""
from typing import Literal, Type

# normal
DATASETS = ["mnist", "emnist_digits", "cifar10", "cifar100"]

# types
DATASETS_LITERAL: Type[
    Literal["mnist", "emnist_digits", "cifar10", "cifar100"]
] = Literal[
    "mnist", "emnist_digits", "cifar10", "cifar100"
]  # type: ignore
