#!/usr/bin/env python
# -*- coding: utf-8 -*-
# mypy: ignore-errors

"""Defines the constants to ensure the consistency and compatibility between the files."""
import enum
from typing import Dict, Any
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


class DATASETS_TYPE(enum.Enum):
    MNIST = "mnist"
    EMNIST_DIGITS = "emnist_digits"
    CIFAR10 = "cifar10"


class OPTIMIZERS_TYPE(enum.Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    ADAMAX = "adamax"
    ADAGRAD = "adagrad"
    ADADALTA = "adadelta"
    ASGD = "asgd"
    LBFGS = "lbfgs"
    NADAM = "nadam"
    RADAM = "radam"
    RMSPROP = "rmsprop"
    RPROP = "rprop"
    SGD = "sgd"
    SPARSEADAM = "sparseadam"


class ACTIVATION_FUNCTIONS_TYPE(enum.Enum):
    TANH = "tanh"
    RELU = "relu"
    LEAKYRELU = "leakyrelu"
    GELU = "gelu"


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
