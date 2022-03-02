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
from torchfl.datamodules.emnist import EMNISTDataModule
from torchfl.models.wrapper.emnist import MNISTEMNIST


def train_model_from_scratch(
    experiment_name: str,
    checkpoint_load_path: Optional[str] = None,
    checkpoint_save_path: str = os.path.join(os.getcwd(), "runs"),
) -> Tuple[nn.Module, Dict[str, float]]:
    """An example wrapper function for training MNIST dataset using PyTorch Lightning trainer and torchfl model and dataloader utilities.

    Args:
        experiment_name (str): Name of the experiment as to be stored in the logs.
        checkpoint_load_path (Optional[str], optional): An optional path to load the model from a given checkpoint. Defaults to None.
        checkpoint_save_path (Optional[str], optional): An optional path to save the trained model, logs, and training related data. Defaults to None.

    Returns:
        Tuple[nn.Module, Dict[str, Tensor]]: Tuple with trained PyTorch model as the first value and the evaluation results as the second value.
    """
    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_save_path, experiment_name),
        accelerator="auto",
        auto_lr_find=True,
        enable_progress_bar=True,
        max_epochs=1,
        num_nodes=torch.cuda.device_count(),
        num_processes=1,
        resume_from_checkpoint=checkpoint_load_path,
        detect_anomaly=True,
        logger=[
            TensorBoardLogger(
                name=experiment_name,
                save_dir=os.path.join(checkpoint_save_path, experiment_name),
            ),
            CSVLogger(save_dir=os.path.join(checkpoint_save_path, experiment_name)),
        ],
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
            DeviceStatsMonitor(),
            ModelSummary(),
            ProgressBar(),
        ],
        progress_bar_refresh_rate=1,
    )
    # prepare the dataset
    datamodule: EMNISTDataModule = EMNISTDataModule(dataset_name="mnist")
    datamodule.prepare_data()
    datamodule.setup()

    # check if the model can be loaded from a given checkpoint
    if checkpoint_load_path and os.path.isfile(checkpoint_load_path):
        model = MNISTEMNIST(
            "mlp", "adam", {"lr": 0.001}, {"img_w": 224, "img_h": 224}
        ).load_from_checkpoint(checkpoint_load_path)
    else:
        pl.seed_everything(42)
        model = MNISTEMNIST("mlp", "adam", {"lr": 0.001}, {"img_w": 224, "img_h": 224})
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
        model = model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,  # type: ignore
        )

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
    return model, result


def train_pretrained_model():
    """Demonstrate a pretrained model training for MNIST."""
    return


def train_finetuned_model():
    """Demonstrate a pretrained model finetuning for MNIST."""
    return


if __name__ == "__main__":
    model, result = train_model_from_scratch("mnist_test")
    print(result)
