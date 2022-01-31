#!/usr/bin/env python
# -*- coding: utf-8 -*-
# mypy: ignore-errors

"""Defines the constants to ensure the consistency and compatibility between the files."""
from typing import Type, Literal, Dict, Any
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
from torch.nn import Tanh, ReLU, LeakyReLU, GELU

# normal
DATASETS = ["mnist", "emnist_digits", "cifar10"]
OPTIMIZERS = [
    "adadelta",
    "adagrad",
    "adam",
    "adamw",
    "sparseadam",
    "adamax",
    "asgd",
    "lbfgs",
    "nadam",
    "radam",
    "rmsprop",
    "rprop",
    "sgd",
]
ACTIVATION_FUNCTIONS = ["tanh", "relu", "leakyrelu", "gelu"]

# type literals
DATASETS_LITERAL: Type[Literal["mnist", "emnist_digits", "cifar10"]] = Literal[
    "mnist", "emnist_digits", "cifar10"
]
OPTIMIZERS_LITERAL: Type[
    Literal[
        "adadelta",
        "adagrad",
        "adam",
        "adamw",
        "sparseadam",
        "adamax",
        "asgd",
        "lbfgs",
        "nadam",
        "radam",
        "rmsprop",
        "rprop",
        "sgd",
    ]
] = Literal[
    "adadelta",
    "adagrad",
    "adam",
    "adamw",
    "sparseadam",
    "adamax",
    "asgd",
    "lbfgs",
    "nadam",
    "radam",
    "rmsprop",
    "rprop",
    "sgd",
]
ACTIVATION_FUNCTIONS_LITERAL: Type[
    Literal["tanh", "relu", "leakyrelu", "gelu"]
] = Literal["tanh", "relu", "leakyrelu", "gelu"]
# mappings
OPTIMIZERS_BY_NAME: Dict[str, Any] = {
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "adam": Adam,
    "adamw": AdamW,
    "sparseadam": SparseAdam,
    "adamax": Adamax,
    "asgd": ASGD,
    "lbfgs": LBFGS,
    "nadam": NAdam,
    "radam": RAdam,
    "rmsprop": RMSprop,
    "rprop": Rprop,
    "sgd": SGD,
}
ACTIVATION_FUNCTIONS_BY_NAME: Dict[str, Any] = {
    "tanh": Tanh,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "gelu": GELU,
}
