#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for EMNIST model wrapper in `torchfl` package."""
import pytest
from torchvision import transforms
from torchfl.datamodules.emnist import EMNISTDataModule
from torchfl.models.wrapper.emnist import (
    BalancedEMNIST,
    ByClassEMNIST,
    ByMergeEMNIST,
    DigitsEMNIST,
    LettersEMNIST,
    MNISTEMNIST,
)
from pytorch_lightning import Trainer

data_transforms = {
    "train_single_channel": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ]
    ),
    "train_three_channel": transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


@pytest.fixture
def emnist_balanced_single_channel_data_module():
    """Fixture for EMNIST single channel data module.

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (balanced).
    """
    global data_transforms
    return EMNISTDataModule(
        dataset_name="balanced",
        train_transforms=data_transforms["train_single_channel"],
    )


@pytest.fixture
def emnist_balanced_three_channel_data_module():
    """Fixture for EMNIST three channel data module.

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (balanced).
    """
    global data_transforms
    return EMNISTDataModule(
        dataset_name="balanced", train_transforms=data_transforms["train_three_channel"]
    )


@pytest.fixture
def emnist_byclass_single_channel_data_module():
    """Fixture for EMNIST single channel data module.

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (byclass).
    """
    global data_transforms
    return EMNISTDataModule(
        dataset_name="byclass", train_transforms=data_transforms["train_single_channel"]
    )


@pytest.fixture
def emnist_byclass_three_channel_data_module():
    """Fixture for EMNIST three channel data module.

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (byclass).
    """
    global data_transforms
    return EMNISTDataModule(
        dataset_name="byclass", train_transforms=data_transforms["train_three_channel"]
    )


@pytest.fixture
def emnist_bymerge_single_channel_data_module():
    """Fixture for EMNIST single channel data module.

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (bymerge).
    """
    global data_transforms
    return EMNISTDataModule(
        dataset_name="bymerge", train_transforms=data_transforms["train_single_channel"]
    )


@pytest.fixture
def emnist_bymerge_three_channel_data_module():
    """Fixture for EMNIST three channel data module.

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (bymerge).
    """
    global data_transforms
    return EMNISTDataModule(
        dataset_name="bymerge", train_transforms=data_transforms["train_three_channel"]
    )


@pytest.fixture
def emnist_digits_single_channel_data_module():
    """Fixture for EMNIST single channel data module.

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (digits).
    """
    global data_transforms
    return EMNISTDataModule(
        dataset_name="digits", train_transforms=data_transforms["train_single_channel"]
    )


@pytest.fixture
def emnist_digits_three_channel_data_module():
    """Fixture for EMNIST three channel data module.

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (digits).
    """
    global data_transforms
    return EMNISTDataModule(
        dataset_name="digits", train_transforms=data_transforms["train_three_channel"]
    )


@pytest.fixture
def emnist_letters_single_channel_data_module():
    """Fixture for EMNIST single channel data module.

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (letters).
    """
    global data_transforms
    return EMNISTDataModule(
        dataset_name="letters", train_transforms=data_transforms["train_single_channel"]
    )


@pytest.fixture
def emnist_letters_three_channel_data_module():
    """Fixture for EMNIST three channel data module.

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (letters).
    """
    global data_transforms
    return EMNISTDataModule(
        dataset_name="letters", train_transforms=data_transforms["train_three_channel"]
    )


@pytest.fixture
def emnist_mnist_single_channel_data_module():
    """Fixture for EMNIST single channel data module.

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (mnist).
    """
    global data_transforms
    return EMNISTDataModule(
        dataset_name="mnist", train_transforms=data_transforms["train_single_channel"]
    )


@pytest.fixture
def emnist_mnist_three_channel_data_module():
    """Fixture for EMNIST three channel data module.

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (mnist).
    """
    global data_transforms
    return EMNISTDataModule(
        dataset_name="mnist", train_transforms=data_transforms["train_three_channel"]
    )


def test_emnist_balanced_single_channel_wrapper(
    emnist_balanced_single_channel_data_module,
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (balanced).
    """
    model = BalancedEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 1}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=emnist_balanced_single_channel_data_module)


def test_emnist_balanced_three_channel_wrapper(
    emnist_balanced_three_channel_data_module,
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (balanced).
    """
    model = BalancedEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=emnist_balanced_three_channel_data_module)


def test_emnist_byclass_single_channel_wrapper(
    emnist_byclass_single_channel_data_module,
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (byclass).
    """
    model = ByClassEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 1}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=emnist_byclass_single_channel_data_module)


def test_emnist_byclass_three_channel_wrapper(emnist_byclass_three_channel_data_module):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (byclass).
    """
    model = ByClassEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=emnist_byclass_three_channel_data_module)


def test_emnist_bymerge_single_channel_wrapper(
    emnist_bymerge_single_channel_data_module,
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (bymerge).
    """
    model = ByMergeEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 1}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=emnist_bymerge_single_channel_data_module)


def test_emnist_bymerge_three_channel_wrapper(emnist_bymerge_three_channel_data_module):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (bymerge).
    """
    model = ByMergeEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=emnist_bymerge_three_channel_data_module)


def test_emnist_digits_single_channel_wrapper(emnist_digits_single_channel_data_module):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (digits).
    """
    model = DigitsEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 1}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=emnist_digits_single_channel_data_module)


def test_emnist_digits_three_channel_wrapper(emnist_digits_three_channel_data_module):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (digits).
    """
    model = DigitsEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=emnist_digits_three_channel_data_module)


def test_emnist_letters_single_channel_wrapper(
    emnist_letters_single_channel_data_module,
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (letters).
    """
    model = LettersEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 1}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=emnist_letters_single_channel_data_module)


def test_emnist_letters_three_channel_wrapper(emnist_letters_three_channel_data_module):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (letters).
    """
    model = LettersEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=emnist_letters_three_channel_data_module)


def test_emnist_mnist_single_channel_wrapper(emnist_mnist_single_channel_data_module):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (mnist).
    """
    model = MNISTEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 1}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=emnist_mnist_single_channel_data_module)


def test_emnist_mnist_three_channel_wrapper(emnist_mnist_three_channel_data_module):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST (mnist).
    """
    model = MNISTEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=emnist_mnist_three_channel_data_module)
