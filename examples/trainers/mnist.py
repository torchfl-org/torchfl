#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""An example script to demonstrate the training of non-federated MNIST dataset using torchfl."""


from typing import Optional, Tuple, List, Dict
import os
import torch.nn as nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    DeviceStatsMonitor,
    ModelSummary,
    ProgressBar,
)
from torchfl.datamodules.emnist import EMNISTDataModule, SUPPORTED_DATASETS_TYPE
from torchfl.models.wrapper.emnist import MNISTEMNIST, EMNIST_MODELS_ENUM
from torchfl.compatibility import OPTIMIZERS_TYPE, TORCHFL_DIR

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)


def train_model_from_scratch(
    experiment_name: str,
    checkpoint_load_path: Optional[str] = None,
    checkpoint_save_path: str = os.path.join(TORCHFL_DIR, "runs"),
) -> Tuple[nn.Module, Dict[str, float]]:
    """An example wrapper function for training MNIST dataset using PyTorch Lightning trainer and torchfl model and dataloader utilities.

    Args:
        experiment_name (str): Name of the experiment as to be stored in the logs.
        checkpoint_load_path (Optional[str], optional): An optional path to load the model from a given checkpoint. Defaults to None.
        checkpoint_save_path (Optional[str], optional): An optional path to save the trained model, logs, and training related data. Defaults to None.

    Returns:
        Tuple[nn.Module, Dict[str, Tensor]]: Tuple with trained PyTorch model as the first value and the evaluation results as the second value.
    """
    ROOT_DIR_PATH: str = os.path.join(checkpoint_save_path, experiment_name)
    trainer = pl.Trainer(
        default_root_dir=ROOT_DIR_PATH,
        accelerator="gpu",
        auto_lr_find=True,
        enable_progress_bar=True,
        max_epochs=10,
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        num_nodes=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        num_processes=1,
        resume_from_checkpoint=checkpoint_load_path,
        detect_anomaly=True,
        logger=[
            TensorBoardLogger(
                name=experiment_name,
                save_dir=ROOT_DIR_PATH,
            ),
            CSVLogger(save_dir=ROOT_DIR_PATH),
        ],
        callbacks=[
            ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
            DeviceStatsMonitor(),
            ModelSummary(),
            ProgressBar(),
        ],
        progress_bar_refresh_rate=1,
    )
    # prepare the dataset
    datamodule: EMNISTDataModule = EMNISTDataModule(
        dataset_name=SUPPORTED_DATASETS_TYPE.MNIST
    )
    datamodule.prepare_data()
    datamodule.setup()

    # check if the model can be loaded from a given checkpoint
    if checkpoint_load_path and os.path.isfile(checkpoint_load_path):
        logging.info("Loading model from the checkpoint at ", checkpoint_load_path)
        model = MNISTEMNIST(
            EMNIST_MODELS_ENUM.MLP,
            OPTIMIZERS_TYPE.ADAM,
            {"lr": 0.001},
            {"img_w": 224, "img_h": 224},
        ).load_from_checkpoint(checkpoint_load_path)
    else:
        pl.seed_everything(42)
        model = MNISTEMNIST(
            EMNIST_MODELS_ENUM.MLP,
            OPTIMIZERS_TYPE.ADAM,
            {"lr": 0.001},
            {"img_w": 224, "img_h": 224},
        )
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())

    # test best model based on the validation and test set
    val_result: List[Dict[str, float]] = trainer.test(
        model, test_dataloaders=datamodule.val_dataloader(), verbose=True
    )
    test_result: List[Dict[str, float]] = trainer.test(
        model, test_dataloaders=datamodule.test_dataloader(), verbose=True
    )
    result: Dict[str, float] = {
        "test": test_result[0]["test_acc"],
        "val": val_result[0]["test_acc"],
    }
    logging.info(
        f"Experiment {experiment_name} completed with results {result} and the log files and checkpoints are stored at {ROOT_DIR_PATH}"
    )
    return model, result
    return model, result


def train_pretrained_model(
    experiment_name: str,
    checkpoint_load_path: Optional[str] = None,
    checkpoint_save_path: str = os.path.join(TORCHFL_DIR, "runs"),
):
    """Demonstrate a pretrained model training (finetuning) for MNIST.

    - Starts training with a pre-trained model and finetunes the weights for all the layers.
    - Quicker than training a model from scratch.

    Args:
        experiment_name (str): Name of the experiment as to be stored in the logs.
        checkpoint_load_path (Optional[str], optional): An optional path to load the model from a given checkpoint. Defaults to None.
        checkpoint_save_path (Optional[str], optional): An optional path to save the trained model, logs, and training related data. Defaults to None.

    Returns:
        Tuple[nn.Module, Dict[str, Tensor]]: Tuple with trained PyTorch model as the first value and the evaluation results as the second value.
    """
    ROOT_DIR_PATH: str = os.path.join(checkpoint_save_path, experiment_name)
    trainer = pl.Trainer(
        default_root_dir=ROOT_DIR_PATH,
        accelerator="auto",
        auto_lr_find=True,
        enable_progress_bar=True,
        max_epochs=1,
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        num_nodes=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        num_processes=1,
        resume_from_checkpoint=checkpoint_load_path,
        detect_anomaly=True,
        logger=[
            TensorBoardLogger(
                name=experiment_name,
                save_dir=ROOT_DIR_PATH,
            ),
            CSVLogger(save_dir=ROOT_DIR_PATH),
        ],
        callbacks=[
            ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
            DeviceStatsMonitor(),
            ModelSummary(),
            ProgressBar(),
        ],
        progress_bar_refresh_rate=1,
    )
    # prepare the dataset
    datamodule: EMNISTDataModule = EMNISTDataModule(
        dataset_name=SUPPORTED_DATASETS_TYPE.MNIST
    )
    datamodule.prepare_data()
    datamodule.setup()

    # check if the model can be loaded from a given checkpoint
    if checkpoint_load_path and os.path.isfile(checkpoint_load_path):
        logging.info("Loading model from the checkpoint at ", checkpoint_load_path)
        model = MNISTEMNIST(
            model_name=EMNIST_MODELS_ENUM.MOBILENETV3SMALL,
            optimizer_name=OPTIMIZERS_TYPE.ADAM,
            optimizer_hparams={"lr": 0.001},
            model_hparams={"pre_trained": True, "feature_extract": False},
        ).load_from_checkpoint(checkpoint_load_path)
    else:
        pl.seed_everything(42)
        model = MNISTEMNIST(
            model_name=EMNIST_MODELS_ENUM.MOBILENETV3SMALL,
            optimizer_name=OPTIMIZERS_TYPE.ADAM,
            optimizer_hparams={"lr": 0.001},
            model_hparams={"pre_trained": True, "feature_extract": False},
        )
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())

    # test best model based on the validation and test set
    val_result: List[Dict[str, float]] = trainer.test(
        model, test_dataloaders=datamodule.val_dataloader(), verbose=True
    )
    test_result: List[Dict[str, float]] = trainer.test(
        model, test_dataloaders=datamodule.test_dataloader(), verbose=True
    )
    result: Dict[str, float] = {
        "test": test_result[0]["test_acc"],
        "val": val_result[0]["test_acc"],
    }
    logging.info(
        f"Experiment {experiment_name} completed with results {result} and the log files and checkpoints are stored at {ROOT_DIR_PATH}"
    )
    return model, result


def train_feature_extraction_model(
    experiment_name: str,
    checkpoint_load_path: Optional[str] = None,
    checkpoint_save_path: str = os.path.join(TORCHFL_DIR, "runs"),
):
    """Demonstrate a pretrained model feature-extraction for MNIST.

    - Starts training with a pre-trained model and only updates the weights for the feature extraction layers.
    - Quickest form of training.

    Args:
        experiment_name (str): Name of the experiment as to be stored in the logs.
        checkpoint_load_path (Optional[str], optional): An optional path to load the model from a given checkpoint. Defaults to None.
        checkpoint_save_path (Optional[str], optional): An optional path to save the trained model, logs, and training related data. Defaults to None.

    Returns:
        Tuple[nn.Module, Dict[str, Tensor]]: Tuple with trained PyTorch model as the first value and the evaluation results as the second value.
    """
    ROOT_DIR_PATH: str = os.path.join(checkpoint_save_path, experiment_name)
    trainer = pl.Trainer(
        default_root_dir=ROOT_DIR_PATH,
        accelerator="auto",
        auto_lr_find=True,
        enable_progress_bar=True,
        max_epochs=1,
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        num_nodes=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        num_processes=1,
        resume_from_checkpoint=checkpoint_load_path,
        detect_anomaly=True,
        logger=[
            TensorBoardLogger(
                name=experiment_name,
                save_dir=ROOT_DIR_PATH,
            ),
            CSVLogger(save_dir=ROOT_DIR_PATH),
        ],
        callbacks=[
            ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
            DeviceStatsMonitor(),
            ModelSummary(),
            ProgressBar(),
        ],
        progress_bar_refresh_rate=1,
    )
    # prepare the dataset
    datamodule: EMNISTDataModule = EMNISTDataModule(
        dataset_name=SUPPORTED_DATASETS_TYPE.MNIST
    )
    datamodule.prepare_data()
    datamodule.setup()

    # check if the model can be loaded from a given checkpoint
    if checkpoint_load_path and os.path.isfile(checkpoint_load_path):
        logging.info("Loading model from the checkpoint at ", checkpoint_load_path)
        model = MNISTEMNIST(
            model_name=EMNIST_MODELS_ENUM.MOBILENETV3SMALL,
            optimizer_name=OPTIMIZERS_TYPE.ADAM,
            optimizer_hparams={"lr": 0.001},
            model_hparams={"pre_trained": True, "feature_extract": True},
        ).load_from_checkpoint(checkpoint_load_path)
    else:
        pl.seed_everything(42)
        model = MNISTEMNIST(
            model_name=EMNIST_MODELS_ENUM.MOBILENETV3SMALL,
            optimizer_name=OPTIMIZERS_TYPE.ADAM,
            optimizer_hparams={"lr": 0.001},
            model_hparams={"pre_trained": True, "feature_extract": True},
        )
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())

    # test best model based on the validation and test set
    val_result: List[Dict[str, float]] = trainer.test(
        model, test_dataloaders=datamodule.val_dataloader(), verbose=True
    )
    test_result: List[Dict[str, float]] = trainer.test(
        model, test_dataloaders=datamodule.test_dataloader(), verbose=True
    )
    result: Dict[str, float] = {
        "test": test_result[0]["test_acc"],
        "val": val_result[0]["test_acc"],
    }
    logging.info(
        f"Experiment {experiment_name} completed with results {result} and the log files and checkpoints are stored at {ROOT_DIR_PATH}"
    )
    return model, result


if __name__ == "__main__":
    model, result = train_model_from_scratch("mnist_mlp_scratch_1")
