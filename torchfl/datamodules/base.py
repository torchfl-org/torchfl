#!/usr/bin/env python

"""Base Data Module class used for defining module operations.

Raises:
    - NotImplementedError: All abstract function are undefined in the base class.
"""
import enum
import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

TORCHFL_DIR: str = os.path.join(Path.home(), ".torchfl")


class BaseDataModule(pl.LightningDataModule, metaclass=ABCMeta):
    """Base Data Module used for defining module operations"""

    def __init__(
        self,
        data_dir: str,
        dataset_name: enum.Enum,
        supported_datasets: set[str],
        train_transforms: transforms.Compose,
        val_transforms: transforms.Compose,
        test_transforms: transforms.Compose,
        predict_transforms: transforms.Compose,
        validation_split: float = 0.1,
        train_batch_size: int = 32,
        validation_batch_size: int = 32,
        test_batch_size: int = 32,
        predict_batch_size: int = 32,
    ):
        super().__init__()
        if not os.path.exists(TORCHFL_DIR):
            os.mkdir(TORCHFL_DIR)
        if dataset_name.value not in supported_datasets:
            raise ValueError(f"{dataset_name}: Not a supported dataset.")
        self.data_dir: str = data_dir
        self.dataset_name: str = dataset_name.value
        self.train_transform: transforms.Compose = train_transforms
        self.val_transform: transforms.Compose = val_transforms
        self.test_transform: transforms.Compose = test_transforms
        self.predict_transform: transforms.Compose = predict_transforms
        self.validation_split: float = validation_split
        self.train_batch_size: int = train_batch_size
        self.validation_batch_size: int = validation_batch_size
        self.test_batch_size: int = test_batch_size
        self.predict_batch_size: int = predict_batch_size
        self.save_hyperparameters()

    @abstractmethod
    def prepare_data(self) -> None:
        """Downloading the data if not already available."""
        raise NotImplementedError(
            f"Subclasses must implement the {self.prepare_data.__name__} method"
        )

    @abstractmethod
    def setup(self, stage: str | None) -> None:
        """Setup before training/testing/validation/prediction using the dataset.

        Args:
            - stage (Optional[str], optional): Current stage of the PyTorch training process used for setup.
        """
        raise NotImplementedError(
            f"Subclasses must implement the {self.setup.__name__} method"
        )

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Training DataLoader wrapper.

        Returns:
            - DataLoader: PyTorch DataLoader object.
        """
        raise NotImplementedError(
            f"Subclasses must implement the {self.train_dataloader.__name__} method"
        )

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Validation DataLoader wrapper.

        Returns:
            - DataLoader: PyTorch DataLoader object.
        """
        raise NotImplementedError(
            f"Subclasses must implement the {self.val_dataloader.__name__} method"
        )

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Test DataLoader wrapper.

        Returns:
            - DataLoader: PyTorch DataLoader object.
        """
        raise NotImplementedError(
            f"Subclasses must implement the {self.test_dataloader.__name__} method"
        )

    @abstractmethod
    def predict_dataloader(self) -> DataLoader:
        """Predict DataLoader object.

        Returns:
            - DataLoader: PyTorch DataLoader object.
        """
        raise NotImplementedError(
            f"Subclasses must implement the {self.predict_dataloader.__name__} method "
        )

    @abstractmethod
    def federated_non_iid_dataloader(
        self,
        num_workers: int,
        workers_batch_size: int,
        niid_factor: int,
    ) -> dict[int, DataLoader]:
        """Loads the training dataset as non-iid split among the workers.

        Args:
            - num_workers (int, optional): number of workers for federated learning.
            - worker_bs (int, optional): batch size of the dataset for workers training locally.
            - niid_factor (int, optional): max number of classes held by each niid agent. lower the number, more measure of non-iidness.

        Returns:
            - Dict[int, DataLoader]: collection of workers as the keys and the PyTorch DataLoader object as values (used for training).
        """
        raise NotImplementedError(
            f"Subclasses must implement the {self.federated_non_iid_dataloader.__name__} method"
        )
