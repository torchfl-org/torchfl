#!/usr/bin/env python
# mypy: ignore-errors

"""Defines the constants to ensure the consistency and compatibility between the files."""
import enum
import os
from pathlib import Path
from typing import Any

from torch.nn import GELU, LeakyReLU, ReLU, Tanh
from torch.optim import (
    ASGD,
    LBFGS,
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SparseAdam,
)

TORCHFL_DIR: str = os.path.join(Path.home(), ".torchfl")
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


class OPTIMIZERS_TYPE(enum.Enum):
    """Enum class for the supported optimizers."""

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
OPTIMIZERS_BY_NAME: dict[str, Any] = {
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
ACTIVATION_FUNCTIONS_BY_NAME: dict[str, Any] = {
    "tanh": Tanh,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "gelu": GELU,
}
