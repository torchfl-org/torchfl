#!/usr/bin/env python

"""PyTorch LightningDataModule for CIFAR(cifar10 and cifar100) dataset.

Raises:
    - ValueError: The given dataset name is not supported. Supported: cifar10, cifar100.

Returns:
    - DatasetSplit: Implementation of PyTorch key-value based Dataset.
    - CIFARDataModule: PyTorch LightningDataModule for CIFAR datasets. Supports iid and non-iid splits.
"""
import enum
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from torchfl.datamodules.base import BaseDataModule

np.random.seed(42)
pl.seed_everything(42)

###################
# Begin Constants #
###################
TORCHFL_DIR: str = os.path.join(Path.home(), ".torchfl")
SUPPORTED_DATASETS: set[str] = {"cifar10", "cifar100"}

DEFAULT_TRANSFORMS: transforms.Compose = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)


class SUPPORTED_DATASETS_TYPE(enum.Enum):
    """Enum for supported datasets."""

    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"


#################
# End Constants #
#################


class DatasetSplit(Dataset):
    """Implementation of PyTorch key-value based Dataset."""

    def __init__(self, dataset: Any, idxs: Iterable[int]) -> None:
        """Constructor

        Args:
            - dataset (Dataset): PyTorch Dataset.
            - idxs (List[int]): collection of indices.
        """
        super().__init__()
        self.dataset: Dataset = dataset
        self.idxs: list[int] = list(idxs)
        all_targets: np.ndarray = (
            np.array(dataset.targets)
            if isinstance(dataset.targets, list)
            else dataset.targets.numpy()
        )
        self.targets: np.ndarray = all_targets[self.idxs]

    def __len__(self) -> int:
        """Overriding the length method.

        Returns:
            - int: length of the collection of indices.
        """
        return len(self.idxs)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Overriding the get method.

        Args:
            - index (int): index for querying.

        Returns:
            - Tuple[Any, Any]: returns the key-value pair as a tuple.
        """
        image, label = self.dataset[self.idxs[index]]
        return image, label


class CIFARDataModule(BaseDataModule):
    """PyTorch LightNingDataModule for CIFAR datasets."""

    def __init__(
        self,
        data_dir: str = os.path.join(TORCHFL_DIR, "data"),
        dataset_name: SUPPORTED_DATASETS_TYPE = SUPPORTED_DATASETS_TYPE.CIFAR10,
        validation_split: float = 0.1,
        train_batch_size: int = 32,
        validation_batch_size: int = 32,
        test_batch_size: int = 32,
        predict_batch_size: int = 32,
        train_transforms: transforms.Compose = DEFAULT_TRANSFORMS,
        val_transforms: transforms.Compose = DEFAULT_TRANSFORMS,
        test_transforms: transforms.Compose = DEFAULT_TRANSFORMS,
        predict_transforms: transforms.Compose = DEFAULT_TRANSFORMS,
    ):
        """Constructor

        Args:
            - data_dir (str, optional): Default directory to download the dataset to. Defaults to os.pardir.
            - dataset_name (str, optional): Name of the dataset to be used. Defaults to "cifar10".
            - validation_split (float, optional): Fraction of training images to be used as validation. Defaults to 0.1.
            - train_batch_size (int, optional): Default batch size of the training data. Defaults to 32.
            - validation_batch_size (int, optional): Default batch size of the validation data. Defaults to 32.
            - test_batch_size (int, optional): Default batch size of the test data. Defaults to 32.
            - predict_batch_size (int, optional): Default batch size of the predict data. Defaults to 32.
            - train_transform (transforms.Compose, optional): Transformations to apply to the training dataset. Defaults to DEFAULT_TRANSFORMS.
            - val_transform (transforms.Compose, optional): Transformations to apply to the validation dataset. Defaults to DEFAULT_TRANSFORMS.
            - test_transform (transforms.Compose, optional): Transformations to apply to the testing dataset. Defaults to DEFAULT_TRANSFORMS.
            - predict_transform (transforms.Compose, optional): Transformations to apply to the prediction dataset. Defaults to DEFAULT_TRANSFORMS.

        Raises:
            - ValueError: The given dataset name is not supported. Supported: cifar10, cifar100.
        """
        super().__init__(
            data_dir=data_dir,
            dataset_name=dataset_name,
            supported_datasets=SUPPORTED_DATASETS,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            predict_transforms=predict_transforms,
            validation_split=validation_split,
            train_batch_size=train_batch_size,
            validation_batch_size=validation_batch_size,
            test_batch_size=test_batch_size,
            predict_batch_size=predict_batch_size,
        )

    def prepare_data(self) -> None:
        """Downloading the data if not already available."""
        if self.dataset_name == "cifar10":
            CIFAR10(self.data_dir, train=True, download=True)
            CIFAR10(self.data_dir, train=False, download=True)
        elif self.dataset_name == "cifar100":
            CIFAR100(self.data_dir, train=True, download=True)
            CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Setup before training/testing/validation/prediction using the dataset.

        Args:
            - stage (Optional[str], optional): Current stage of the PyTorch training process used for setup. Defaults to None.
        """
        total_images: int | None = None
        num_validation_images: int | None = None
        num_training_images: int | None = None
        if self.dataset_name == "cifar10":
            total_images = len(
                CIFAR10(self.data_dir, train=True, download=True)
            )
            num_validation_images = int(total_images * self.validation_split)
            num_training_images = total_images - num_validation_images
            if (stage == "fit") or (not stage):
                self.cifar_train_full = CIFAR10(
                    self.data_dir,
                    train=True,
                    download=True,
                    transform=self.train_transform,
                )
                self.cifar_train, self.cifar_val = random_split(
                    self.cifar_train_full,
                    [num_training_images, num_validation_images],
                )
            if (stage == "test") or (not stage):
                self.cifar_test = CIFAR10(
                    self.data_dir,
                    train=False,
                    download=True,
                    transform=self.test_transform,
                )
            if (stage == "predict") or (not stage):
                self.cifar_predict = CIFAR10(
                    self.data_dir,
                    train=False,
                    download=True,
                    transform=self.predict_transform,
                )
        elif self.dataset_name == "cifar100":
            total_images = len(
                CIFAR100(self.data_dir, train=True, download=True)
            )
            num_validation_images = int(total_images * self.validation_split)
            num_training_images = total_images - num_validation_images
            if (stage == "fit") or (not stage):
                self.cifar_train_full = CIFAR100(
                    self.data_dir,
                    train=True,
                    download=True,
                    transform=self.train_transform,
                )
                self.cifar_train, self.cifar_val = random_split(
                    self.cifar_train_full,
                    [num_training_images, num_validation_images],
                )
            if (stage == "test") or (not stage):
                self.cifar_test = CIFAR100(
                    self.data_dir, train=False, transform=self.test_transform
                )
            if (stage == "predict") or (not stage):
                self.cifar_predict = CIFAR100(
                    self.data_dir,
                    train=False,
                    transform=self.predict_transform,
                )

    def train_dataloader(self) -> DataLoader:
        """Training DataLoader wrapper.

        Returns:
            - DataLoader: PyTorch DataLoader object.
        """
        return DataLoader(
            self.cifar_train,
            batch_size=self.train_batch_size,
            num_workers=os.cpu_count() or 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation DataLoader wrapper.

        Returns:
            - DataLoader: PyTorch DataLoader object.
        """
        return DataLoader(
            self.cifar_val,
            batch_size=self.validation_batch_size,
            num_workers=os.cpu_count() or 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Test DataLoader wrapper.

        Returns:
            - DataLoader: PyTorch DataLoader object.
        """
        return DataLoader(
            self.cifar_test,
            batch_size=self.test_batch_size,
            num_workers=os.cpu_count() or 0,
        )

    def predict_dataloader(self) -> DataLoader:
        """Predict DataLoader object.

        Returns:
            - DataLoader: PyTorch DataLoader object.
        """
        return DataLoader(
            self.cifar_predict,
            batch_size=self.predict_batch_size,
            num_workers=os.cpu_count() or 0,
        )

    def federated_iid_dataloader(
        self, num_workers: int = 10, workers_batch_size: int = 10
    ) -> dict[int, DataLoader]:
        """Loads the training dataset as iid split among the workers.

        Args:
            - num_workers (int, optional): number of workers for federated learning. Defaults to 10.
            - worker_bs (int, optional): batch size of the dataset for workers training locally. Defaults to 10.

        Returns:
            - Dict[int, DataLoader]: collection of workers as the keys and the PyTorch DataLoader object as values (used for training).
        """
        items: int = len(self.cifar_train_full) // num_workers
        distribution: np.ndarray = np.random.randint(
            low=0, high=len(self.cifar_train_full), size=(num_workers, items)
        )
        federated: dict[int, DataLoader] = {}
        for i in range(len(distribution)):
            federated[i] = DataLoader(
                DatasetSplit(self.cifar_train_full, distribution[i]),
                batch_size=workers_batch_size,
                shuffle=True,
                num_workers=os.cpu_count() or 0,
            )
        return federated

    def federated_non_iid_dataloader(
        self,
        num_workers: int = 10,
        workers_batch_size: int = 10,
        niid_factor: int = 2,
    ) -> dict[int, DataLoader]:
        """Loads the training dataset as non-iid split among the workers.

        Args:
            - num_workers (int, optional): number of workers for federated learning. Defaults to 10.
            - worker_bs (int, optional): batch size of the dataset for workers training locally. Defaults to 10.
            - niid_factor (int, optional): max number of classes held by each niid agent. lower the number, more measure of non-iidness. Defaults to 2.

        Returns:
            - Dict[int, DataLoader]: collection of workers as the keys and the PyTorch DataLoader object as values (used for training).
        """
        shards: int = num_workers * niid_factor
        items: int = len(self.cifar_train_full) // shards
        idx_shard: list[int] = list(range(shards))
        classes: np.ndarray = (
            np.array(self.cifar_train_full.targets)
            if isinstance(self.cifar_train_full.targets, list)
            else self.cifar_train_full.targets.numpy()
        )

        idxs_labels: np.ndarray = np.vstack(
            (np.arange(len(self.cifar_train_full)), classes)
        )
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs: np.ndarray = idxs_labels[0, :]
        distribution: dict[int, np.ndarray] = {
            i: np.array([], dtype="int64") for i in range(num_workers)
        }
        while idx_shard:
            for i in range(num_workers):
                rand_set: set[int] = set(
                    np.random.choice(idx_shard, niid_factor, replace=False)
                )
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    distribution[i] = np.concatenate(
                        (
                            distribution[i],
                            idxs[rand * items : (rand + 1) * items],
                        ),
                        axis=0,
                    )
        federated: dict[int, DataLoader] = {}
        for i in distribution:
            federated[i] = DataLoader(
                DatasetSplit(self.cifar_train_full, distribution[i]),
                batch_size=workers_batch_size,
                shuffle=True,
                num_workers=os.cpu_count() or 0,
            )
        return federated
