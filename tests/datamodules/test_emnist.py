#!/usr/bin/env python

"""Tests for EMNIST PyTorch LightningDataModule module in `torchfl` package."""
from collections import Counter

import pytest

from torchfl.datamodules.emnist import (
    SUPPORTED_DATASETS_TYPE,
    EMNISTDataModule,
)


@pytest.fixture()
def emnist_balanced_data_module():
    """Fixture for EMNIST PyTorch LightningDataModule

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST
    """
    return EMNISTDataModule(dataset_name=SUPPORTED_DATASETS_TYPE.BALANCED)


@pytest.fixture()
def emnist_byclass_data_module():
    """Fixture for EMNIST PyTorch LightningDataModule

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST
    """
    return EMNISTDataModule(dataset_name=SUPPORTED_DATASETS_TYPE.BYCLASS)


@pytest.fixture()
def emnist_bymerge_data_module():
    """Fixture for EMNIST PyTorch LightningDataModule

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    return EMNISTDataModule(dataset_name=SUPPORTED_DATASETS_TYPE.BYMERGE)


@pytest.fixture()
def emnist_digits_data_module():
    """Fixture for EMNIST PyTorch LightningDataModule

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    return EMNISTDataModule(dataset_name=SUPPORTED_DATASETS_TYPE.DIGITS)


@pytest.fixture()
def emnist_letters_data_module():
    """Fixture for EMNIST PyTorch LightningDataModule

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    return EMNISTDataModule(dataset_name=SUPPORTED_DATASETS_TYPE.LETTERS)


@pytest.fixture()
def emnist_mnist_data_module():
    """Fixture for EMNIST PyTorch LightningDataModule

    Returns:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    return EMNISTDataModule(dataset_name=SUPPORTED_DATASETS_TYPE.MNIST)


############
# Balanced #
############


@pytest.mark.datamodules_emnist_balanced()
def test_emnist_balanced_train_val_split(emnist_balanced_data_module):
    """Testing the EMNIST dataset train and validation split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_balanced_data_module.prepare_data()
    emnist_balanced_data_module.setup(stage="fit")
    train_dataloader = emnist_balanced_data_module.train_dataloader()
    val_dataloader = emnist_balanced_data_module.val_dataloader()
    assert len(train_dataloader.dataset) == 101520
    assert len(val_dataloader.dataset) == 11280


@pytest.mark.datamodules_emnist_balanced()
def test_emnist_balanced_test_split(emnist_balanced_data_module):
    """Testing the EMNIST dataset test split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_balanced_data_module.prepare_data()
    emnist_balanced_data_module.setup(stage="test")
    test_dataloader = emnist_balanced_data_module.test_dataloader()
    assert len(test_dataloader.dataset) == 18800


@pytest.mark.datamodules_emnist_balanced()
def test_emnist_balanced_prediction_split(emnist_balanced_data_module):
    """Testing the EMNIST dataset prediction split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_balanced_data_module.prepare_data()
    emnist_balanced_data_module.setup(stage="predict")
    predict_dataloader = emnist_balanced_data_module.predict_dataloader()
    assert len(predict_dataloader.dataset) == 18800


@pytest.mark.datamodules_emnist_balanced()
def test_emnist_balanced_federated_iid_split(emnist_balanced_data_module):
    """Testing the EMNIST dataset federated iid split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_balanced_data_module.prepare_data()
    emnist_balanced_data_module.setup(stage="fit")
    dataloader = emnist_balanced_data_module.federated_iid_dataloader()
    assert len(dataloader.keys()) == 10
    assert len(dataloader[0].dataset) == 11280
    frequency = Counter(list(dataloader[0].dataset.targets))
    assert len(frequency.keys()) == 47


@pytest.mark.datamodules_emnist_balanced()
def test_emnist_balanced_federated_non_iid_split(emnist_balanced_data_module):
    """Testing the EMNIST dataset federated non iid split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_balanced_data_module.prepare_data()
    emnist_balanced_data_module.setup(stage="fit")
    dataloader = emnist_balanced_data_module.federated_non_iid_dataloader()
    assert len(dataloader.keys()) == 10
    all_freq = []
    for i in range(10):
        assert len(dataloader[i].dataset) == 11280
        frequency = Counter(list(dataloader[i].dataset.targets))
        all_freq.append(len(frequency.keys()))
    assert max(all_freq) == 8


############
# By Class #
############


@pytest.mark.datamodules_emnist_byclass()
def test_emnist_byclass_train_val_split(emnist_byclass_data_module):
    """Testing the EMNIST dataset train and validation split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_byclass_data_module.prepare_data()
    emnist_byclass_data_module.setup(stage="fit")
    train_dataloader = emnist_byclass_data_module.train_dataloader()
    val_dataloader = emnist_byclass_data_module.val_dataloader()
    assert len(train_dataloader.dataset) == 628139
    assert len(val_dataloader.dataset) == 69793


@pytest.mark.datamodules_emnist_byclass()
def test_emnist_byclass_test_split(emnist_byclass_data_module):
    """Testing the EMNIST dataset test split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_byclass_data_module.prepare_data()
    emnist_byclass_data_module.setup(stage="test")
    test_dataloader = emnist_byclass_data_module.test_dataloader()
    assert len(test_dataloader.dataset) == 116323


@pytest.mark.datamodules_emnist_byclass()
def test_emnist_byclass_prediction_split(emnist_byclass_data_module):
    """Testing the EMNIST dataset prediction split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_byclass_data_module.prepare_data()
    emnist_byclass_data_module.setup(stage="predict")
    predict_dataloader = emnist_byclass_data_module.predict_dataloader()
    assert len(predict_dataloader.dataset) == 116323


@pytest.mark.datamodules_emnist_byclass()
def test_emnist_byclass_federated_iid_split(emnist_byclass_data_module):
    """Testing the EMNIST dataset federated iid split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_byclass_data_module.prepare_data()
    emnist_byclass_data_module.setup(stage="fit")
    dataloader = emnist_byclass_data_module.federated_iid_dataloader()
    assert len(dataloader.keys()) == 10
    assert len(dataloader[0].dataset) == 69793
    frequency = Counter(list(dataloader[0].dataset.targets))
    assert len(frequency.keys()) == 62


@pytest.mark.datamodules_emnist_byclass()
def test_emnist_byclass_federated_non_iid_split(emnist_byclass_data_module):
    """Testing the EMNIST dataset federated non iid split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_byclass_data_module.prepare_data()
    emnist_byclass_data_module.setup(stage="fit")
    dataloader = emnist_byclass_data_module.federated_non_iid_dataloader()
    assert len(dataloader.keys()) == 10
    all_freq = []
    for i in range(10):
        assert len(dataloader[i].dataset) == 69792
        frequency = Counter(list(dataloader[i].dataset.targets))
        all_freq.append(len(frequency.keys()))
    assert max(all_freq) == 16


############
# By Merge #
############


@pytest.mark.datamodules_emnist_bymerge()
def test_emnist_bymerge_train_val_split(emnist_bymerge_data_module):
    """Testing the EMNIST dataset train and validation split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_bymerge_data_module.prepare_data()
    emnist_bymerge_data_module.setup(stage="fit")
    train_dataloader = emnist_bymerge_data_module.train_dataloader()
    val_dataloader = emnist_bymerge_data_module.val_dataloader()
    assert len(train_dataloader.dataset) == 628139
    assert len(val_dataloader.dataset) == 69793


@pytest.mark.datamodules_emnist_bymerge()
def test_emnist_bymerge_test_split(emnist_bymerge_data_module):
    """Testing the EMNIST dataset test split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_bymerge_data_module.prepare_data()
    emnist_bymerge_data_module.setup(stage="test")
    test_dataloader = emnist_bymerge_data_module.test_dataloader()
    assert len(test_dataloader.dataset) == 116323


@pytest.mark.datamodules_emnist_bymerge()
def test_emnist_bymerge_prediction_split(emnist_bymerge_data_module):
    """Testing the EMNIST dataset prediction split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_bymerge_data_module.prepare_data()
    emnist_bymerge_data_module.setup(stage="predict")
    predict_dataloader = emnist_bymerge_data_module.predict_dataloader()
    assert len(predict_dataloader.dataset) == 116323


@pytest.mark.datamodules_emnist_bymerge()
def test_emnist_bymerge_federated_iid_split(emnist_bymerge_data_module):
    """Testing the EMNIST dataset federated iid split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_bymerge_data_module.prepare_data()
    emnist_bymerge_data_module.setup(stage="fit")
    dataloader = emnist_bymerge_data_module.federated_iid_dataloader()
    assert len(dataloader.keys()) == 10
    assert len(dataloader[0].dataset) == 69793
    frequency = Counter(list(dataloader[0].dataset.targets))
    assert len(frequency.keys()) == 47


@pytest.mark.datamodules_emnist_bymerge()
def test_emnist_bymerge_federated_non_iid_split(emnist_bymerge_data_module):
    """Testing the EMNIST dataset federated non iid split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_bymerge_data_module.prepare_data()
    emnist_bymerge_data_module.setup(stage="fit")
    dataloader = emnist_bymerge_data_module.federated_non_iid_dataloader()
    assert len(dataloader.keys()) == 10
    all_freq = []
    for i in range(10):
        assert len(dataloader[i].dataset) == 69792
        frequency = Counter(list(dataloader[i].dataset.targets))
        all_freq.append(len(frequency.keys()))
    assert max(all_freq) == 12


##########
# Digits #
##########


@pytest.mark.datamodules_emnist_digits()
def test_emnist_digits_train_val_split(emnist_digits_data_module):
    """Testing the EMNIST dataset train and validation split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_digits_data_module.prepare_data()
    emnist_digits_data_module.setup(stage="fit")
    train_dataloader = emnist_digits_data_module.train_dataloader()
    val_dataloader = emnist_digits_data_module.val_dataloader()
    assert len(train_dataloader.dataset) == 216000
    assert len(val_dataloader.dataset) == 24000


@pytest.mark.datamodules_emnist_digits()
def test_emnist_digits_test_split(emnist_digits_data_module):
    """Testing the EMNIST dataset test split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_digits_data_module.prepare_data()
    emnist_digits_data_module.setup(stage="test")
    test_dataloader = emnist_digits_data_module.test_dataloader()
    assert len(test_dataloader.dataset) == 40000


@pytest.mark.datamodules_emnist_digits()
def test_emnist_digits_prediction_split(emnist_digits_data_module):
    """Testing the EMNIST dataset prediction split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_digits_data_module.prepare_data()
    emnist_digits_data_module.setup(stage="predict")
    predict_dataloader = emnist_digits_data_module.predict_dataloader()
    assert len(predict_dataloader.dataset) == 40000


@pytest.mark.datamodules_emnist_digits()
def test_emnist_digits_federated_iid_split(emnist_digits_data_module):
    """Testing the EMNIST dataset federated iid split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_digits_data_module.prepare_data()
    emnist_digits_data_module.setup(stage="fit")
    dataloader = emnist_digits_data_module.federated_iid_dataloader()
    assert len(dataloader.keys()) == 10
    assert len(dataloader[0].dataset) == 24000
    frequency = Counter(list(dataloader[0].dataset.targets))
    assert len(frequency.keys()) == 10


@pytest.mark.datamodules_emnist_digits()
def test_emnist_digits_federated_non_iid_split(emnist_digits_data_module):
    """Testing the EMNIST dataset federated non iid split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_digits_data_module.prepare_data()
    emnist_digits_data_module.setup(stage="fit")
    dataloader = emnist_digits_data_module.federated_non_iid_dataloader()
    assert len(dataloader.keys()) == 10
    all_freq = []
    for i in range(10):
        assert len(dataloader[i].dataset) == 24000
        frequency = Counter(list(dataloader[i].dataset.targets))
        all_freq.append(len(frequency.keys()))
    assert max(all_freq) == 2


###########
# Letters #
###########


@pytest.mark.datamodules_emnist_letters()
def test_emnist_letters_train_val_split(emnist_letters_data_module):
    """Testing the EMNIST dataset train and validation split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_letters_data_module.prepare_data()
    emnist_letters_data_module.setup(stage="fit")
    train_dataloader = emnist_letters_data_module.train_dataloader()
    val_dataloader = emnist_letters_data_module.val_dataloader()
    assert len(train_dataloader.dataset) == 112320
    assert len(val_dataloader.dataset) == 12480


@pytest.mark.datamodules_emnist_letters()
def test_emnist_letters_test_split(emnist_letters_data_module):
    """Testing the EMNIST dataset test split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_letters_data_module.prepare_data()
    emnist_letters_data_module.setup(stage="test")
    test_dataloader = emnist_letters_data_module.test_dataloader()
    assert len(test_dataloader.dataset) == 20800


@pytest.mark.datamodules_emnist_letters()
def test_emnist_letters_prediction_split(emnist_letters_data_module):
    """Testing the EMNIST dataset prediction split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_letters_data_module.prepare_data()
    emnist_letters_data_module.setup(stage="predict")
    predict_dataloader = emnist_letters_data_module.predict_dataloader()
    assert len(predict_dataloader.dataset) == 20800


@pytest.mark.datamodules_emnist_letters()
def test_emnist_letters_federated_iid_split(emnist_letters_data_module):
    """Testing the EMNIST dataset federated iid split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_letters_data_module.prepare_data()
    emnist_letters_data_module.setup(stage="fit")
    dataloader = emnist_letters_data_module.federated_iid_dataloader()
    assert len(dataloader.keys()) == 10
    assert len(dataloader[0].dataset) == 12480
    frequency = Counter(list(dataloader[0].dataset.targets))
    assert len(frequency.keys()) == 26


@pytest.mark.datamodules_emnist_letters()
def test_emnist_letters_federated_non_iid_split(emnist_letters_data_module):
    """Testing the EMNIST dataset federated non iid split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_letters_data_module.prepare_data()
    emnist_letters_data_module.setup(stage="fit")
    dataloader = emnist_letters_data_module.federated_non_iid_dataloader()
    assert len(dataloader.keys()) == 10
    all_freq = []
    for i in range(10):
        assert len(dataloader[i].dataset) == 12480
        frequency = Counter(list(dataloader[i].dataset.targets))
        all_freq.append(len(frequency.keys()))
    assert max(all_freq) == 5


#########
# MNIST #
#########


@pytest.mark.datamodules_emnist_mnist()
def test_emnist_mnist_train_val_split(emnist_mnist_data_module):
    """Testing the EMNIST dataset train and validation split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_mnist_data_module.prepare_data()
    emnist_mnist_data_module.setup(stage="fit")
    train_dataloader = emnist_mnist_data_module.train_dataloader()
    val_dataloader = emnist_mnist_data_module.val_dataloader()
    assert len(train_dataloader.dataset) == 54000
    assert len(val_dataloader.dataset) == 6000


@pytest.mark.datamodules_emnist_mnist()
def test_emnist_mnist_test_split(emnist_mnist_data_module):
    """Testing the EMNIST dataset test split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_mnist_data_module.prepare_data()
    emnist_mnist_data_module.setup(stage="test")
    test_dataloader = emnist_mnist_data_module.test_dataloader()
    assert len(test_dataloader.dataset) == 10000


@pytest.mark.datamodules_emnist_mnist()
def test_emnist_mnist_prediction_split(emnist_mnist_data_module):
    """Testing the EMNIST dataset prediction split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_mnist_data_module.prepare_data()
    emnist_mnist_data_module.setup(stage="predict")
    predict_dataloader = emnist_mnist_data_module.predict_dataloader()
    assert len(predict_dataloader.dataset) == 10000


@pytest.mark.datamodules_emnist_mnist()
def test_emnist_mnist_federated_iid_split(emnist_mnist_data_module):
    """Testing the EMNIST dataset federated iid split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_mnist_data_module.prepare_data()
    emnist_mnist_data_module.setup(stage="fit")
    dataloader = emnist_mnist_data_module.federated_iid_dataloader()
    assert len(dataloader.keys()) == 10
    assert len(dataloader[0].dataset) == 6000
    frequency = Counter(list(dataloader[0].dataset.targets))
    assert len(frequency.keys()) == 10


@pytest.mark.datamodules_emnist_mnist()
def test_emnist_mnist_federated_non_iid_split(emnist_mnist_data_module):
    """Testing the EMNIST dataset federated non iid split with PyTorch Lightning wrapper.

    Args:
        EMNISTDataModule: PyTorch LightningDataModule for EMNIST.
    """
    emnist_mnist_data_module.prepare_data()
    emnist_mnist_data_module.setup(stage="fit")
    dataloader = emnist_mnist_data_module.federated_non_iid_dataloader()
    assert len(dataloader.keys()) == 10
    all_freq = []
    for i in range(10):
        assert len(dataloader[i].dataset) == 6000
        frequency = Counter(list(dataloader[i].dataset.targets))
        all_freq.append(len(frequency.keys()))
    assert max(all_freq) == 2
