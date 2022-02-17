#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for EMNIST model wrapper in `torchfl` package."""
import pytest
from torchvision import datasets, transforms
import os
from torchfl.models.wrapper.emnist import (
    BalancedEMNIST,
    ByClassEMNIST,
    ByMergeEMNIST,
    DigitsEMNIST,
    LettersEMNIST,
    MNISTEMNIST,
)
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

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
def emnist_balanced_train_single_channel_loader():
    """Fixture for EMNIST single channel train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        split="balanced",
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def emnist_balanced_test_single_channel_loader():
    """Fixture for EMNIST single channel test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=False,
        split="balanced",
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def emnist_balanced_train_three_channel_loader():
    """Fixture for EMNIST three channel train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        split="balanced",
        download=True,
        transform=data_transforms["train_three_channel"],
    )


@pytest.fixture
def emnist_balanced_test_three_channel_loader():
    """Fixture for EMNIST three channel test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=False,
        split="balanced",
        download=True,
        transform=data_transforms["train_three_channel"],
    )


@pytest.fixture
def emnist_byclass_train_single_channel_loader():
    """Fixture for EMNIST single channel train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        split="byclass",
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def emnist_byclass_test_single_channel_loader():
    """Fixture for EMNIST single channel test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=False,
        split="byclass",
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def emnist_byclass_train_three_channel_loader():
    """Fixture for EMNIST three channel train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        split="byclass",
        download=True,
        transform=data_transforms["train_three_channel"],
    )


@pytest.fixture
def emnist_byclass_test_three_channel_loader():
    """Fixture for EMNIST three channel test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=False,
        split="byclass",
        download=True,
        transform=data_transforms["train_three_channel"],
    )


@pytest.fixture
def emnist_bymerge_train_single_channel_loader():
    """Fixture for EMNIST single channel train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        split="bymerge",
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def emnist_bymerge_test_single_channel_loader():
    """Fixture for EMNIST single channel test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=False,
        split="bymerge",
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def emnist_bymerge_train_three_channel_loader():
    """Fixture for EMNIST three channel train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        split="bymerge",
        download=True,
        transform=data_transforms["train_three_channel"],
    )


@pytest.fixture
def emnist_bymerge_test_three_channel_loader():
    """Fixture for EMNIST three channel test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=False,
        split="bymerge",
        download=True,
        transform=data_transforms["train_three_channel"],
    )


@pytest.fixture
def emnist_digits_train_single_channel_loader():
    """Fixture for EMNIST single channel train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        split="digits",
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def emnist_digits_test_single_channel_loader():
    """Fixture for EMNIST single channel test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=False,
        split="digits",
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def emnist_digits_train_three_channel_loader():
    """Fixture for EMNIST three channel train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        split="digits",
        download=True,
        transform=data_transforms["train_three_channel"],
    )


@pytest.fixture
def emnist_digits_test_three_channel_loader():
    """Fixture for EMNIST three channel test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=False,
        split="digits",
        download=True,
        transform=data_transforms["train_three_channel"],
    )


@pytest.fixture
def emnist_letters_train_single_channel_loader():
    """Fixture for EMNIST single channel train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        split="letters",
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def emnist_letters_test_single_channel_loader():
    """Fixture for EMNIST single channel test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=False,
        split="letters",
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def emnist_letters_train_three_channel_loader():
    """Fixture for EMNIST three channel train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        split="letters",
        download=True,
        transform=data_transforms["train_three_channel"],
    )


@pytest.fixture
def emnist_letters_test_three_channel_loader():
    """Fixture for EMNIST three channel test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=False,
        split="letters",
        download=True,
        transform=data_transforms["train_three_channel"],
    )


@pytest.fixture
def emnist_mnist_train_single_channel_loader():
    """Fixture for EMNIST single channel train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        split="mnist",
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def emnist_mnist_test_single_channel_loader():
    """Fixture for EMNIST single channel test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=False,
        split="mnist",
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def emnist_mnist_train_three_channel_loader():
    """Fixture for EMNIST three channel train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        split="mnist",
        download=True,
        transform=data_transforms["train_three_channel"],
    )


@pytest.fixture
def emnist_mnist_test_three_channel_loader():
    """Fixture for EMNIST three channel test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=False,
        split="mnist",
        download=True,
        transform=data_transforms["train_three_channel"],
    )


def test_emnist_balanced_single_channel_wrapper(
    emnist_balanced_train_single_channel_loader,
    emnist_balanced_test_single_channel_loader,
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        emnist_balanced_train_single_channel_loader (Dataset): PyTorch Dataset object.
        emnist_balanced_test_single_channel_loader (Dataset): PyTorch Dataset object.
    """
    model = BalancedEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 1}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=emnist_balanced_train_single_channel_loader, batch_size=1),
        DataLoader(dataset=emnist_balanced_test_single_channel_loader, batch_size=1),
    )


def test_emnist_balanced_three_channel_wrapper(
    emnist_balanced_train_three_channel_loader,
    emnist_balanced_test_three_channel_loader,
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        emnist_balanced_train_three_channel_loader (Dataset): PyTorch Dataset object.
        emnist_balanced_test_three_channel_loader (Dataset): PyTorch Dataset object.
    """
    model = BalancedEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=emnist_balanced_train_three_channel_loader, batch_size=1),
        DataLoader(dataset=emnist_balanced_test_three_channel_loader, batch_size=1),
    )


def test_emnist_byclass_single_channel_wrapper(
    emnist_byclass_train_single_channel_loader,
    emnist_byclass_test_single_channel_loader,
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        emnist_byclass_train_single_channel_loader (Dataset): PyTorch Dataset object.
        emnist_byclass_test_single_channel_loader (Dataset): PyTorch Dataset object.
    """
    model = ByClassEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 1}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=emnist_byclass_train_single_channel_loader, batch_size=1),
        DataLoader(dataset=emnist_byclass_test_single_channel_loader, batch_size=1),
    )


def test_emnist_byclass_three_channel_wrapper(
    emnist_byclass_train_three_channel_loader, emnist_byclass_test_three_channel_loader
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        emnist_byclass_train_three_channel_loader (Dataset): PyTorch Dataset object.
        emnist_byclass_test_three_channel_loader (Dataset): PyTorch Dataset object.
    """
    model = ByClassEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=emnist_byclass_train_three_channel_loader, batch_size=1),
        DataLoader(dataset=emnist_byclass_test_three_channel_loader, batch_size=1),
    )


def test_emnist_bymerge_single_channel_wrapper(
    emnist_bymerge_train_single_channel_loader,
    emnist_bymerge_test_single_channel_loader,
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        emnist_bymerge_train_single_channel_loader (Dataset): PyTorch Dataset object.
        emnist_bymerge_test_single_channel_loader (Dataset): PyTorch Dataset object.
    """
    model = ByMergeEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 1}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=emnist_bymerge_train_single_channel_loader, batch_size=1),
        DataLoader(dataset=emnist_bymerge_test_single_channel_loader, batch_size=1),
    )


def test_emnist_bymerge_three_channel_wrapper(
    emnist_bymerge_train_three_channel_loader, emnist_bymerge_test_three_channel_loader
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        emnist_bymerge_train_three_channel_loader (Dataset): PyTorch Dataset object.
        emnist_bymerge_test_three_channel_loader (Dataset): PyTorch Dataset object.
    """
    model = ByMergeEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=emnist_bymerge_train_three_channel_loader, batch_size=1),
        DataLoader(dataset=emnist_bymerge_test_three_channel_loader, batch_size=1),
    )


def test_emnist_digits_single_channel_wrapper(
    emnist_digits_train_single_channel_loader, emnist_digits_test_single_channel_loader
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        emnist_digits_train_single_channel_loader (Dataset): PyTorch Dataset object.
        emnist_digits_test_single_channel_loader (Dataset): PyTorch Dataset object.
    """
    model = DigitsEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 1}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=emnist_digits_train_single_channel_loader, batch_size=1),
        DataLoader(dataset=emnist_digits_test_single_channel_loader, batch_size=1),
    )


def test_emnist_digits_three_channel_wrapper(
    emnist_digits_train_three_channel_loader, emnist_digits_test_three_channel_loader
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        emnist_digits_train_three_channel_loader (Dataset): PyTorch Dataset object.
        emnist_digits_test_three_channel_loader (Dataset): PyTorch Dataset object.
    """
    model = DigitsEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=emnist_digits_train_three_channel_loader, batch_size=1),
        DataLoader(dataset=emnist_digits_test_three_channel_loader, batch_size=1),
    )


def test_emnist_letters_single_channel_wrapper(
    emnist_letters_train_single_channel_loader,
    emnist_letters_test_single_channel_loader,
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        emnist_letters_train_single_channel_loader (Dataset): PyTorch Dataset object.
        emnist_letters_test_single_channel_loader (Dataset): PyTorch Dataset object.
    """
    model = LettersEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 1}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=emnist_letters_train_single_channel_loader, batch_size=1),
        DataLoader(dataset=emnist_letters_test_single_channel_loader, batch_size=1),
    )


def test_emnist_letters_three_channel_wrapper(
    emnist_letters_train_three_channel_loader, emnist_letters_test_three_channel_loader
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        emnist_letters_train_three_channel_loader (Dataset): PyTorch Dataset object.
        emnist_letters_test_three_channel_loader (Dataset): PyTorch Dataset object.
    """
    model = LettersEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=emnist_letters_train_three_channel_loader, batch_size=1),
        DataLoader(dataset=emnist_letters_test_three_channel_loader, batch_size=1),
    )


def test_emnist_mnist_single_channel_wrapper(
    emnist_mnist_train_single_channel_loader, emnist_mnist_test_single_channel_loader
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        emnist_mnist_train_single_channel_loader (Dataset): PyTorch Dataset object.
        emnist_mnist_test_single_channel_loader (Dataset): PyTorch Dataset object.
    """
    model = MNISTEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 1}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=emnist_mnist_train_single_channel_loader, batch_size=1),
        DataLoader(dataset=emnist_mnist_test_single_channel_loader, batch_size=1),
    )


def test_emnist_mnist_three_channel_wrapper(
    emnist_mnist_train_three_channel_loader, emnist_mnist_test_three_channel_loader
):
    """Testing the EMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        emnist_mnist_train_three_channel_loader (Dataset): PyTorch Dataset object.
        emnist_mnist_test_three_channel_loader (Dataset): PyTorch Dataset object.
    """
    model = MNISTEMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=emnist_mnist_train_three_channel_loader, batch_size=1),
        DataLoader(dataset=emnist_mnist_test_three_channel_loader, batch_size=1),
    )
