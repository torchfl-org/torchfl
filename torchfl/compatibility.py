#!/usr/bin/env python
# -*- coding: utf-8 -*-
# mypy: ignore-errors

"""Defines the constants to ensure the consistency and compatibility between the files."""
from typing import Type, Literal, Union
from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    AdamW,
    SparseAdam,
    Adamax,
    ASGD,
    LBFGS,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
)

# normal
DATASETS = ["mnist", "emnist_digits", "cifar10"]

# types
DATASETS_LITERAL: Type[Literal["mnist", "emnist_digits", "cifar10"]] = Literal[
    "mnist", "emnist_digits", "cifar10"
]
OPTIMIZERS: Type[
    Union[
        Adadelta,
        Adagrad,
        Adam,
        AdamW,
        SparseAdam,
        Adamax,
        ASGD,
        LBFGS,
        NAdam,
        RAdam,
        RMSprop,
        Rprop,
        SGD,
    ]
] = Union[
    Adadelta,
    Adagrad,
    Adam,
    AdamW,
    SparseAdam,
    Adamax,
    ASGD,
    LBFGS,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
]
