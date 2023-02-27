#!/usr/bin/env python

"""Types used within the datamodules utilities."""

import enum

from torchfl.datamodules.cifar import (
    SUPPORTED_DATASETS_TYPE as CIFAR_DATASETS_TYPE,
)
from torchfl.datamodules.cifar import CIFARDataModule
from torchfl.datamodules.emnist import (
    SUPPORTED_DATASETS_TYPE as EMNIST_DATASETS_TYPE,
)
from torchfl.datamodules.emnist import EMNISTDataModule
from torchfl.datamodules.fashionmnist import (
    SUPPORTED_DATASETS_TYPE as FASHIONMNIST_DATASETS_TYPE,
)
from torchfl.datamodules.fashionmnist import FashionMNISTDataModule
from torchfl.utils import _get_enum_values

EMNIST_DATASETS: list[str] = _get_enum_values(EMNIST_DATASETS_TYPE)
CIFAR_DATASETS: list[str] = _get_enum_values(CIFAR_DATASETS_TYPE)
FASHIONMNIST_DATASETS: list[str] = _get_enum_values(FASHIONMNIST_DATASETS_TYPE)

DATASET_GROUPS_MAP: dict[str, list[str]] = {
    "emnist": EMNIST_DATASETS,
    "cifar": CIFAR_DATASETS,
    "fashionmnist": FASHIONMNIST_DATASETS,
}


class DatasetGroupsEnum(enum.Enum):
    EMNIST = EMNISTDataModule
    CIFAR = CIFARDataModule
    FASHIONMNIST = FashionMNISTDataModule


DatasetGroupsType = EMNISTDataModule | CIFARDataModule | FashionMNISTDataModule
