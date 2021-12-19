#!/usr/bin/env python
# -*- coding: utf-8 -*-
# mypy: ignore-errors

"""Prepares the given torchvision datasets for federated learning.

Raises:
    ValueError: The given name of the dataset is not currently supported.

Returns:
    DatasetSplit: Implementation of PyTorch key-value based Dataset.
    FLDataLoader: Prepares the torchvision datasets for federated learning. Supports iid and non-iid splits.
"""
import os
from typing import Any, Dict, Iterable, List, Set, Tuple
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from torchfl.compatibility import DATASETS_LITERAL

np.random.seed(42)


class DatasetSplit(Dataset):
    """Implementation of PyTorch key-value based Dataset."""

    def __init__(self, dataset: Dataset, idxs: Iterable[int]) -> None:
        """Constructor

        Args:
            dataset (Dataset): PyTorch Dataset.
            idxs (Iterable[int]): collection of indices.
        """
        self.dataset: Dataset = dataset
        self.idxs: Iterable[int] = list(idxs)

    def __len__(self) -> int:
        """Overriding the length method.

        Returns:
            int: length of the collection of indices.
        """
        return len(self.idxs)  # type: ignore

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Overriding the get method.

        Args:
            index (int): index for querying.

        Returns:
            Tuple[Any, Any]: returns the key-value pair as a tuple.
        """
        image, label = self.dataset[self.idxs[index]]  # type: ignore
        return image, label


class FLDataLoader:
    """Prepares the torchvision datasets for federated learning. Supports iid and non-iid splits."""

    def __init__(
        self,
        num_workers: int = 10,
        worker_bs: int = 10,
        iid: bool = True,
        niid_factor: int = 2,
        dataset: DATASETS_LITERAL = "mnist",  # type: ignore
        test_bs: int = 128,
    ) -> None:
        """Constructor

        Args:
            num_workers (int, optional): number of workers for federated learning. Defaults to 10.
            worker_bs (int, optional): batch size of the dataset for workers training locally. Defaults to 10.
            worker_ep (int, optional): number of epochs for the workers training locally. Defaults to 5.
            iid (bool, optional): whether the dataset follows iid distribution or not. Defaults to True.
            niid_factor (int, optional): max number of classes held by each niid agent. lower the number, more measure of non-iidness. Defaults to 2.
            dataset ([mnist, emnist_digits, cifar10], optional): name of the dataset to be used. Defaults to "mnist".
            test_bs (int, optional): batch size used for the testing dataset. Defaults to 128.
        """
        self.num_workers: int = num_workers
        self.worker_bs: int = worker_bs
        self.iid: bool = iid
        self.niid_factor: int = niid_factor
        self.dataset: DATASETS_LITERAL = dataset  # type: ignore
        self.test_bs: int = test_bs

    @staticmethod
    def __load_dataset__(
        name: DATASETS_LITERAL, training: bool
    ) -> Dataset:  # type: ignore
        """Helper method used to load the PyTorch Dataset with a provided name.

        Args:
            name ([mnist, emnist_digits, cifar10]): name of the dataset to be loaded.
            training (bool): if the dataset needs to be used for training or testing.

        Raises:
            ValueError: the given name is not currently supported.

        Returns:
            Dataset: PyTorch Dataset object.
        """
        root = os.path.join(os.pardir, "data")
        if name.lower() == "mnist":
            return datasets.MNIST(
                root,
                train=training,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            )
        elif name.lower() == "emnist_digits":  # type: ignore
            return datasets.EMNIST(
                root,
                split="digits",
                train=training,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            )

        elif name.lower() == "cifar10":  # type: ignore
            return datasets.CIFAR10(
                root,
                train=training,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                ),
            )

        else:
            raise ValueError(
                f"name: the dataset is not currently supported. Found: {name}"
            )

    def train_iid(self) -> Dict[int, DataLoader]:
        """Loads the training dataset as iid split among the workers.

        Returns:
            Dict[int, DataLoader]: collection of workers as the keys and the PyTorch DataLoader object as values (used for training).
        """
        dataset: Dataset = self.__load_dataset__(self.dataset, True)
        items: int = len(dataset) // self.num_workers
        distribution: np.ndarray = np.random.randint(
            low=0, high=len(dataset), size=(self.num_workers, items)
        )
        federated: Dict[int, Dataset] = dict()
        for i in range(len(distribution)):
            federated[i] = DataLoader(
                DatasetSplit(dataset, distribution[i]),
                batch_size=self.worker_bs,
                shuffle=True,
            )
        return federated

    def train_non_iid(self) -> Dict[int, DataLoader]:
        """Loads the training dataset as non-iid split among the workers.

        Returns:
            Dict[int, DataLoader]: collection of workers as the keys and the PyTorch DataLoader object as values (used for training).
        """
        dataset: Dataset = self.__load_dataset__(self.dataset, True)
        shards: int = self.num_workers * self.niid_factor
        items: int = len(dataset) // shards
        idx_shard: List[int] = list(range(shards))
        classes: np.ndarray = np.array([])
        if isinstance(dataset.targets, list):
            classes = np.array(dataset.targets)
        else:
            classes = dataset.targets.numpy()

        idxs_labels: np.ndarray = np.vstack((np.arange(len(dataset)), classes))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs: np.ndarray = idxs_labels[0, :]
        distribution: Dict[int, np.ndarray] = {
            i: np.array([], dtype="int64") for i in range(self.num_workers)
        }

        while idx_shard:
            for i in range(self.num_workers):
                rand_set: Set[int] = set(
                    np.random.choice(idx_shard, self.niid_factor, replace=False)
                )
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    distribution[i] = np.concatenate(
                        (distribution[i], idxs[rand * items : (rand + 1) * items]),
                        axis=0,
                    )
        federated: Dict[int, Dataset] = dict()
        for i in distribution:
            federated[i] = DataLoader(
                DatasetSplit(dataset, distribution[i]),
                batch_size=self.worker_bs,
                shuffle=True,
            )
        return federated

    def test(self) -> Dataset:
        """Loads the dataset for testing.

        Returns:
            Dataset: PyTorch Dataset object.
        """
        return self.__load_dataset__(self.dataset, False)


# driver for testing
if __name__ == "__main__":
    from cli import cli_parser

    args = cli_parser()
    loader = FLDataLoader()
    print(loader.train_non_iid())
