#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""An example script to demonstrate the training of non-federated CIFAR-10 dataset using torchfl."""

from typing import Optional, Tuple, List, Dict
import os
import torch.nn as nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    DeviceStatsMonitor,
    ModelSummary,
    RichProgressBar,
    Timer,
)
from torchfl.datamodules.cifar import CIFARDataModule, SUPPORTED_DATASETS_TYPE
from torchfl.models.wrapper.cifar import CIFAR10, CIFAR_MODELS_ENUM
from torchfl.compatibility import OPTIMIZERS_TYPE, TORCHFL_DIR

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)


def train_model_from_scratch(
    experiment_name: str,
    checkpoint_load_path: Optional[str] = None,
    checkpoint_save_path: str = os.path.join(TORCHFL_DIR, "runs"),
) -> Tuple[nn.Module, Dict[str, float]]:
    """An example wrapper function for training CIFAR10 dataset using PyTorch Lightning trainer and torchfl model and dataloader utilities.
    Args:
        experiment_name (str): Name of the experiment as to be stored in the logs.
        checkpoint_load_path (Optional[str], optional): An optional path to load the model from a given checkpoint. Defaults to None.
        checkpoint_save_path (Optional[str], optional): An optional path to save the trained model, logs, and training related data. Defaults to None.
    Returns:
        Tuple[nn.Module, Dict[str, Tensor]]: Tuple with trained PyTorch model as the first value and the evaluation results as the second value.
    """
    """An example wrapper function for training CIFAR10 dataset using PyTorch Lightning trainer and torchfl model and dataloader utilities.
    Args:
        experiment_name (str): Name of the experiment as to be stored in the logs.
        checkpoint_load_path (Optional[str], optional): An optional path to load the model from a given checkpoint. Defaults to None.
        checkpoint_save_path (Optional[str], optional): An optional path to save the trained model, logs, and training related data. Defaults to None.
    Returns:
        Tuple[nn.Module, Dict[str, Tensor]]: Tuple with trained PyTorch model as the first value and the evaluation results as the second value.
    """
    ROOT_DIR_PATH: str = os.path.join(checkpoint_save_path, experiment_name)
    tb_logger: TensorBoardLogger = TensorBoardLogger(
        name=experiment_name, save_dir=ROOT_DIR_PATH
    )
    csv_logger: CSVLogger = CSVLogger(save_dir=ROOT_DIR_PATH)
    pt_profiler: PyTorchProfiler = PyTorchProfiler(
        dirpath=ROOT_DIR_PATH,
        group_by_input_shapes=True,
        record_module_names=True,
        export_to_chrome=True,
    )
    # prepare the dataset
    datamodule: CIFARDataModule = CIFARDataModule(
        dataset_name=SUPPORTED_DATASETS_TYPE.CIFAR10,
        train_batch_size=128,
    )
    datamodule.prepare_data()
    datamodule.setup()

    # check if the model can be loaded from a given checkpoint
    if checkpoint_load_path and os.path.isfile(checkpoint_load_path):
        logging.info("Loading model from the checkpoint at ", checkpoint_load_path)
        model = CIFAR10(
            CIFAR_MODELS_ENUM.RESNET152,
            OPTIMIZERS_TYPE.ADAM,
            {"lr": 0.001},
            {"pre_trained": False, "feature_extract": False},
        ).load_from_checkpoint(checkpoint_load_path)
    else:
        pl.seed_everything(42)
        model = CIFAR10(
            CIFAR_MODELS_ENUM.RESNET152,
            OPTIMIZERS_TYPE.ADAM,
            {"lr": 0.001},
            {"pre_trained": False, "feature_extract": False},
        )

    tb_logger.log_graph(model, datamodule.train_dataloader().dataset[0][0])
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
            tb_logger,
            csv_logger,
        ],
        callbacks=[
            # ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
            DeviceStatsMonitor(),
            ModelSummary(),
            RichProgressBar(leave=True),
            Timer(),
        ],
        enable_checkpointing=False,  # temp fix for MemoryError on GPU because of limited cache
    )
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())

    # test best model based on the validation and test set
    val_result: List[Dict[str, float]] = trainer.test(
        model, dataloaders=datamodule.val_dataloader(), verbose=True
    )
    test_result: List[Dict[str, float]] = trainer.test(
        model, dataloaders=datamodule.test_dataloader(), verbose=True
    )
    result: Dict[str, float] = {
        "test": test_result[0]["test_acc"],
        "val": val_result[0]["test_acc"],
    }
    logging.info(
        f"Experiment {experiment_name} completed with results {result} and the log files and checkpoints are stored at {ROOT_DIR_PATH}"
    )
    return model, result


def train_pretrained_model(
    experiment_name: str,
    checkpoint_load_path: Optional[str] = None,
    checkpoint_save_path: str = os.path.join(TORCHFL_DIR, "runs"),
) -> Tuple[nn.Module, Dict[str, float]]:
    """An example wrapper function for training CIFAR10 dataset using PyTorch Lightning trainer and torchfl model and dataloader utilities.
    Args:
        experiment_name (str): Name of the experiment as to be stored in the logs.
        checkpoint_load_path (Optional[str], optional): An optional path to load the model from a given checkpoint. Defaults to None.
        checkpoint_save_path (Optional[str], optional): An optional path to save the trained model, logs, and training related data. Defaults to None.
    Returns:
        Tuple[nn.Module, Dict[str, Tensor]]: Tuple with trained PyTorch model as the first value and the evaluation results as the second value.
    """
    ROOT_DIR_PATH: str = os.path.join(checkpoint_save_path, experiment_name)
    tb_logger: TensorBoardLogger = TensorBoardLogger(
        name=experiment_name, save_dir=ROOT_DIR_PATH
    )
    csv_logger: CSVLogger = CSVLogger(save_dir=ROOT_DIR_PATH)
    pt_profiler: PyTorchProfiler = PyTorchProfiler(
        dirpath=ROOT_DIR_PATH,
        group_by_input_shapes=True,
        record_module_names=True,
        export_to_chrome=True,
    )
    # prepare the dataset
    datamodule: CIFARDataModule = CIFARDataModule(
        dataset_name=SUPPORTED_DATASETS_TYPE.CIFAR10,
        train_batch_size=32,
    )
    datamodule.prepare_data()
    datamodule.setup()

    # check if the model can be loaded from a given checkpoint
    if checkpoint_load_path and os.path.isfile(checkpoint_load_path):
        logging.info("Loading model from the checkpoint at ", checkpoint_load_path)
        model = CIFAR10(
            CIFAR_MODELS_ENUM.RESNET152,
            OPTIMIZERS_TYPE.ADAM,
            {"lr": 0.001},
            {"pre_trained": True, "feature_extract": False},
        ).load_from_checkpoint(checkpoint_load_path)
    else:
        pl.seed_everything(42)
        model = CIFAR10(
            CIFAR_MODELS_ENUM.RESNET152,
            OPTIMIZERS_TYPE.ADAM,
            {"lr": 0.001},
            {"pre_trained": True, "feature_extract": False},
        )

    tb_logger.log_graph(model, datamodule.train_dataloader().dataset[0][0])
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
            tb_logger,
            csv_logger,
        ],
        callbacks=[
            # ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
            DeviceStatsMonitor(),
            ModelSummary(),
            RichProgressBar(leave=True),
            Timer(),
        ],
        enable_checkpointing=False,  # temp fix for MemoryError on GPU because of limited cache
    )
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())

    # test best model based on the validation and test set
    val_result: List[Dict[str, float]] = trainer.test(
        model, dataloaders=datamodule.val_dataloader(), verbose=True
    )
    test_result: List[Dict[str, float]] = trainer.test(
        model, dataloaders=datamodule.test_dataloader(), verbose=True
    )
    result: Dict[str, float] = {
        "test": test_result[0]["test_acc"],
        "val": val_result[0]["test_acc"],
    }
    logging.info(
        f"Experiment {experiment_name} completed with results {result} and the log files and checkpoints are stored at {ROOT_DIR_PATH}"
    )
    return model, result


def train_feature_extracted_model(
    experiment_name: str,
    checkpoint_load_path: Optional[str] = None,
    checkpoint_save_path: str = os.path.join(TORCHFL_DIR, "runs"),
) -> Tuple[nn.Module, Dict[str, float]]:
    """An example wrapper function for training CIFAR10 dataset using PyTorch Lightning trainer and torchfl model and dataloader utilities.
    Args:
        experiment_name (str): Name of the experiment as to be stored in the logs.
        checkpoint_load_path (Optional[str], optional): An optional path to load the model from a given checkpoint. Defaults to None.
        checkpoint_save_path (Optional[str], optional): An optional path to save the trained model, logs, and training related data. Defaults to None.
    Returns:
        Tuple[nn.Module, Dict[str, Tensor]]: Tuple with trained PyTorch model as the first value and the evaluation results as the second value.
    """
    ROOT_DIR_PATH: str = os.path.join(checkpoint_save_path, experiment_name)
    tb_logger: TensorBoardLogger = TensorBoardLogger(
        name=experiment_name, save_dir=ROOT_DIR_PATH
    )
    csv_logger: CSVLogger = CSVLogger(save_dir=ROOT_DIR_PATH)
    pt_profiler: PyTorchProfiler = PyTorchProfiler(
        dirpath=ROOT_DIR_PATH,
        group_by_input_shapes=True,
        record_module_names=True,
        export_to_chrome=True,
    )
    # prepare the dataset
    datamodule: CIFARDataModule = CIFARDataModule(
        dataset_name=SUPPORTED_DATASETS_TYPE.CIFAR10,
        train_batch_size=128,
    )
    datamodule.prepare_data()
    datamodule.setup()

    # check if the model can be loaded from a given checkpoint
    if checkpoint_load_path and os.path.isfile(checkpoint_load_path):
        logging.info("Loading model from the checkpoint at ", checkpoint_load_path)
        model = CIFAR10(
            CIFAR_MODELS_ENUM.RESNET152,
            OPTIMIZERS_TYPE.ADAM,
            {"lr": 0.001},
            {"pre_trained": True, "feature_extract": True},
        ).load_from_checkpoint(checkpoint_load_path)
    else:
        pl.seed_everything(42)
        model = CIFAR10(
            CIFAR_MODELS_ENUM.RESNET152,
            OPTIMIZERS_TYPE.ADAM,
            {"lr": 0.001},
            {"pre_trained": True, "feature_extract": True},
        )

    tb_logger.log_graph(model, datamodule.train_dataloader().dataset[0][0])
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
            tb_logger,
            csv_logger,
        ],
        callbacks=[
            # ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
            DeviceStatsMonitor(),
            ModelSummary(),
            RichProgressBar(leave=True),
            Timer(),
        ],
        enable_checkpointing=False,  # temp fix for MemoryError on GPU because of limited cache
    )
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())

    # test best model based on the validation and test set
    val_result: List[Dict[str, float]] = trainer.test(
        model, dataloaders=datamodule.val_dataloader(), verbose=True
    )
    test_result: List[Dict[str, float]] = trainer.test(
        model, dataloaders=datamodule.test_dataloader(), verbose=True
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
    model, result = train_pretrained_model("cifar10_resnet152_pretrained")
    logging.info(result)
