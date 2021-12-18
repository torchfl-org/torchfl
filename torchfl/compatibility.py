from typing import Literal, Type

# normal
DATASETS = ["mnist", "emnist_digits", "cifar10", "cifar100"]

# types
DATASETS_LITERAL: Type[
    Literal["mnist", "emnist_digits", "cifar10", "cifar100"]
] = Literal["mnist", "emnist_digits", "cifar10", "cifar100"]
