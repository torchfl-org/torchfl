#!/usr/bin/env python

"""Contains the PyTorch Lightning wrapper module for FashionMNIST dataset."""
import enum
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor, optim

from torchfl.compatibility import OPTIMIZERS_BY_NAME, OPTIMIZERS_TYPE
from torchfl.federated.fl_params import FLParams
from torchfl.models.core.fashionmnist.alexnet import (
    AlexNet as FashionMNISTAlexNet,
)
from torchfl.models.core.fashionmnist.densenet import (
    DenseNet121 as FashionMNISTDenseNet121,
)
from torchfl.models.core.fashionmnist.densenet import (
    DenseNet161 as FashionMNISTDenseNet161,
)
from torchfl.models.core.fashionmnist.densenet import (
    DenseNet169 as FashionMNISTDenseNet169,
)
from torchfl.models.core.fashionmnist.densenet import (
    DenseNet201 as FashionMNISTDenseNet201,
)
from torchfl.models.core.fashionmnist.lenet import LeNet as FashionMNISTLeNet
from torchfl.models.core.fashionmnist.mlp import MLP as FashionMNISTMLP
from torchfl.models.core.fashionmnist.mobilenet import (
    MobileNetV2 as FashionMNISTMobileNetV2,
)
from torchfl.models.core.fashionmnist.mobilenet import (
    MobileNetV3Large as FashionMNISTMobileNetV3Large,
)
from torchfl.models.core.fashionmnist.mobilenet import (
    MobileNetV3Small as FashionMNISTMobileNetV3Small,
)
from torchfl.models.core.fashionmnist.resnet import (
    ResNet18 as FashionMNISTResNet18,
)
from torchfl.models.core.fashionmnist.resnet import (
    ResNet34 as FashionMNISTResNet34,
)
from torchfl.models.core.fashionmnist.resnet import (
    ResNet50 as FashionMNISTResNet50,
)
from torchfl.models.core.fashionmnist.resnet import (
    ResNet101 as FashionMNISTResNet101,
)
from torchfl.models.core.fashionmnist.resnet import (
    ResNet152 as FashionMNISTResNet152,
)
from torchfl.models.core.fashionmnist.resnet import (
    ResNext50_32X4D as FashionMNISTResNext50_32X4D,
)
from torchfl.models.core.fashionmnist.resnet import (
    ResNext101_32X8D as FashionMNISTResNext101_32X8D,
)
from torchfl.models.core.fashionmnist.resnet import (
    WideResNet50_2 as FashionMNISTWideResNet50_2,
)
from torchfl.models.core.fashionmnist.resnet import (
    WideResNet101_2 as FashionMNISTWideResNet101_2,
)
from torchfl.models.core.fashionmnist.shufflenetv2 import (
    ShuffleNetv2_x0_5 as FashionMNISTShuffleNetv2_x0_5,
)
from torchfl.models.core.fashionmnist.shufflenetv2 import (
    ShuffleNetv2_x1_0 as FashionMNISTShuffleNetv2_x1_0,
)
from torchfl.models.core.fashionmnist.shufflenetv2 import (
    ShuffleNetv2_x1_5 as FashionMNISTShuffleNetv2_x1_5,
)
from torchfl.models.core.fashionmnist.shufflenetv2 import (
    ShuffleNetv2_x2_0 as FashionMNISTShuffleNetv2_x2_0,
)
from torchfl.models.core.fashionmnist.squeezenet import (
    SqueezeNet1_0 as FashionMNISTSqueezeNet1_0,
)
from torchfl.models.core.fashionmnist.squeezenet import (
    SqueezeNet1_1 as FashionMNISTSqueezeNet1_1,
)
from torchfl.models.core.fashionmnist.vgg import VGG11 as FashionMNISTVGG11
from torchfl.models.core.fashionmnist.vgg import (
    VGG11_BN as FashionMNISTVGG11_BN,
)
from torchfl.models.core.fashionmnist.vgg import VGG13 as FashionMNISTVGG13
from torchfl.models.core.fashionmnist.vgg import (
    VGG13_BN as FashionMNISTVGG13_BN,
)
from torchfl.models.core.fashionmnist.vgg import VGG16 as FashionMNISTVGG16
from torchfl.models.core.fashionmnist.vgg import (
    VGG16_BN as FashionMNISTVGG16_BN,
)
from torchfl.models.core.fashionmnist.vgg import VGG19 as FashionMNISTVGG19
from torchfl.models.core.fashionmnist.vgg import (
    VGG19_BN as FashionMNISTVGG19_BN,
)

pl.seed_everything(42)

###############
# Begin Utils #
###############

FASHIONMNIST_MODELS: list[str] = [
    "alexnet",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "lenet",
    "mlp",
    "mobilenetv2",
    "mobilenetv3small",
    "mobilenetv3large",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wideresnet50_2",
    "wideresnet101_2",
    "shufflenetv2_x0_5",
    "shufflenetv2_x1_0",
    "shufflenetv2_x1_5",
    "shufflenetv2_x2_0",
    "squeezenet1_0",
    "squeezenet1_1",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]


class FASHIONMNIST_MODELS_ENUM(enum.Enum):
    """Enum for supported FashionMNIST models."""

    ALEXNET = "alexnet"
    DENSENET121 = "densenet121"
    DENSENET161 = "densenet161"
    DENSENET169 = "densenet169"
    DENSENET201 = "densenet201"
    LENET = "lenet"
    MLP = "mlp"
    MOBILENETV2 = "mobilenetv2"
    MOBILENETV3SMALL = "mobilenetv3small"
    MOBILENETV3LARGE = "mobilenetv3large"
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    RESNET152 = "resnet152"
    RESNEXT50_32X4D = "resnext50_32x4d"
    RESNEXT101_32X8D = "resnext101_32x8d"
    WIDERESNET50_2 = "wideresnet50_2"
    WIDERESNET101_2 = "wideresnet101_2"
    SHUFFLENETV2_X0_5 = "shufflenetv2_x0_5"
    SHUFFLENETV2_X1_0 = "shufflenetv2_x1_0"
    SHUFFLENETV2_X1_5 = "shufflenetv2_x1_5"
    SHUFFLENETV2_X2_0 = "shufflenetv2_x2_0"
    SQUEEZENET1_0 = "squeezenet1_0"
    SQUEEZENET1_1 = "squeezenet1_1"
    VGG11 = "vgg11"
    VGG11_BN = "vgg11_bn"
    VGG13 = "vgg13"
    VGG13_BN = "vgg13_bn"
    VGG16 = "vgg16"
    VGG16_BN = "vgg16_bn"
    VGG19 = "vgg19"
    VGG19_BN = "vgg19_bn"


FASHIONMNIST_MODEL_TYPE = (
    type[FashionMNISTAlexNet]
    | type[FashionMNISTDenseNet121]
    | type[FashionMNISTDenseNet161]
    | type[FashionMNISTDenseNet169]
    | type[FashionMNISTDenseNet201]
    | type[FashionMNISTLeNet]
    | type[FashionMNISTMLP]
    | type[FashionMNISTMobileNetV2]
    | type[FashionMNISTMobileNetV3Large]
    | type[FashionMNISTMobileNetV3Small]
    | type[FashionMNISTResNet18]
    | type[FashionMNISTResNet34]
    | type[FashionMNISTResNet50]
    | type[FashionMNISTResNet101]
    | type[FashionMNISTResNet152]
    | type[FashionMNISTResNext50_32X4D]
    | type[FashionMNISTResNext101_32X8D]
    | type[FashionMNISTWideResNet50_2]
    | type[FashionMNISTWideResNet101_2]
    | type[FashionMNISTShuffleNetv2_x0_5]
    | type[FashionMNISTShuffleNetv2_x1_0]
    | type[FashionMNISTShuffleNetv2_x1_5]
    | type[FashionMNISTShuffleNetv2_x2_0]
    | type[FashionMNISTSqueezeNet1_0]
    | type[FashionMNISTSqueezeNet1_1]
    | type[FashionMNISTVGG11]
    | type[FashionMNISTVGG11_BN]
    | type[FashionMNISTVGG13]
    | type[FashionMNISTVGG13_BN]
    | type[FashionMNISTVGG16]
    | type[FashionMNISTVGG16_BN]
    | type[FashionMNISTVGG19]
    | type[FashionMNISTVGG19_BN]
)

FASHIONMNIST_MODELS_MAPPING: dict[str, FASHIONMNIST_MODEL_TYPE] = {
    "alexnet": FashionMNISTAlexNet,
    "densenet121": FashionMNISTDenseNet121,
    "densenet161": FashionMNISTDenseNet161,
    "densenet169": FashionMNISTDenseNet169,
    "densenet201": FashionMNISTDenseNet201,
    "lenet": FashionMNISTLeNet,
    "mlp": FashionMNISTMLP,
    "mobilenetv2": FashionMNISTMobileNetV2,
    "mobilenetv3small": FashionMNISTMobileNetV3Small,
    "mobilenetv3large": FashionMNISTMobileNetV3Large,
    "resnet18": FashionMNISTResNet18,
    "resnet34": FashionMNISTResNet34,
    "resnet50": FashionMNISTResNet50,
    "resnet101": FashionMNISTResNet101,
    "resnet152": FashionMNISTResNet152,
    "resnext50_32x4d": FashionMNISTResNext50_32X4D,
    "resnext101_32x8d": FashionMNISTResNext101_32X8D,
    "wideresnet50_2": FashionMNISTWideResNet50_2,
    "wideresnet101_2": FashionMNISTWideResNet101_2,
    "shufflenetv2_x0_5": FashionMNISTShuffleNetv2_x0_5,
    "shufflenetv2_x1_0": FashionMNISTShuffleNetv2_x1_0,
    "shufflenetv2_x1_5": FashionMNISTShuffleNetv2_x1_5,
    "shufflenetv2_x2_0": FashionMNISTShuffleNetv2_x2_0,
    "squeezenet1_0": FashionMNISTSqueezeNet1_0,
    "squeezenet1_1": FashionMNISTSqueezeNet1_1,
    "vgg11": FashionMNISTVGG11,
    "vgg11_bn": FashionMNISTVGG11_BN,
    "vgg13": FashionMNISTVGG13,
    "vgg13_bn": FashionMNISTVGG13_BN,
    "vgg16": FashionMNISTVGG16,
    "vgg16_bn": FashionMNISTVGG16_BN,
    "vgg19": FashionMNISTVGG19,
    "vgg19_bn": FashionMNISTVGG19_BN,
}


def create_model(
    dataset_name: str,
    model_name: str,
    model_hparams: dict[str, Any] | None = None,
) -> FASHIONMNIST_MODEL_TYPE:
    """Helper function to create a model from the available options.

    Args:
        - dataset_name (str): Name of the dataset.
        - model_name (str): Name of the model for the dataset.
        - model_hparams (Optional[Dict[str, Any]]): Hyperparameters for the model. Defaults to None.

    Returns:
        - nn.Module: PyTorch model definition.

    Raises:
        - ValueError: Unsupported dataset name or model.
    """
    if dataset_name == "fashionmnist":
        if model_name not in FASHIONMNIST_MODELS_MAPPING:
            raise ValueError(
                f"{model_name}: Invalid model name. Not supported for this dataset."
            )
        else:
            return (
                FASHIONMNIST_MODELS_MAPPING[model_name](**model_hparams)
                if model_hparams
                else FASHIONMNIST_MODELS_MAPPING[model_name]()
            )

    else:
        raise ValueError(
            f"{dataset_name}: Invalid dataset name. Not a supported dataset."
        )


###############
# End Utils #
###############


class FashionMNIST(pl.LightningModule):
    """PyTorch Lightning wrapper for FashionMNIST dataset."""

    def __init__(
        self,
        model_name: FASHIONMNIST_MODELS_ENUM,
        optimizer_name: OPTIMIZERS_TYPE,
        optimizer_hparams: dict[str, Any],
        model_hparams: dict[str, Any] | None = None,
        fl_hparams: FLParams | None = None,
    ) -> None:
        """Default constructor.

        Args:
            - model_name (str): Name of the model to be used. Only choose from the available models.
            - optimizer_name (str): Name of optimizer to be used. Only choose from the available models.
            - optimizer_hparams(Dict[str, Any]): Hyperparameters to initialize the optimizer.
            - model_hparams (Optional[Dict[str, Any]], optional): Optional override the default model hparams. Defaults to None.
            - fl_hparams (Optional[FLParams], optional): Optional override the default federated learning hparams. Defaults to None.
        """
        super().__init__()
        self.model = torch.compile(
            create_model(
                dataset_name="fashionmnist",
                model_name=model_name.value,
                model_hparams=model_hparams,
            )
        )
        self.fl_hparams: dict[str, Any] | None = (
            fl_hparams.as_dict() if fl_hparams else None
        )
        combined_hparams: dict[str, Any] = {
            "model_hparams": vars(self.model.hparams),
            "optimizer_hparams": {
                "optimizer_name": optimizer_name,
                "optimizer_fn": OPTIMIZERS_BY_NAME[optimizer_name.value],
                "config": optimizer_hparams,
            },
            "fl_hparams": vars(fl_hparams.as_simple_namespace())
            if fl_hparams
            else {},
        }
        self.save_hyperparameters(combined_hparams)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, imgs: Tensor) -> Any:  # type: ignore
        """Forward propagation

        Args:
            - imgs (Tensor): Images for forward propagation.

        Returns:
            - Tensor: PyTorch Tensor generated from forward propagation.
        """
        return self.model(imgs)

    def configure_optimizers(self):
        """Configuring the optimizer and scheduler for training process."""
        optimizer_fn = self.hparams.optimizer_hparams["optimizer_fn"]
        optimizer: optimizer_fn = optimizer_fn(
            self.parameters(), **self.hparams.optimizer_hparams["config"]
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(  # type: ignore
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:  # type: ignore
        """Training step

        Args:
            - batch (Tuple[Tensor, Tensor]): Batch of the training data.
            - batch_idx (int): Index of the batch.

        Returns:
            - Tensor: PyTorch Tensor to call ".backward" on
        """
        imgs, labels = batch
        preds: Tensor = self.model(imgs)
        loss: Tensor = self.loss_module(preds, labels)
        acc: Tensor = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the fl-related metrics if fl_hparams is not None
        if self.fl_hparams is not None:
            self.log(
                f"{self.fl_hparams.get('experiment_name', 'default')}_train_loss",
                loss,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{self.fl_hparams.get('experiment_name', 'default')}_train_acc",
                acc,
                on_step=False,
                on_epoch=True,
            )
        else:
            # Logs the accuracy per epoch (weighted average over batches)
            self.log("batch_idx", batch_idx)
            self.log("train_acc", acc, on_step=False, on_epoch=True)
            self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(  # type: ignore
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:  # type: ignore
        """Validation step

        Args:
            - batch (Tuple[Tensor, Tensor]): Batch of the validation data.
            - batch_idx (int): Index of the batch.
        """
        imgs, labels = batch
        preds: Tensor = self.model(imgs)
        loss: Tensor = self.loss_module(preds, labels)
        preds = self.model(imgs).argmax(dim=-1)
        acc: Tensor = (labels == preds).float().mean()

        # Logs the fl-related metrics if fl_hparams is not None
        if self.fl_hparams is not None:
            self.log(
                f"{self.fl_hparams.get('experiment_name', 'default')}_test_loss",
                loss,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{self.fl_hparams.get('experiment_name', 'default')}_test_acc",
                acc,
                on_step=False,
                on_epoch=True,
            )
        else:
            # By default logs it per epoch (weighted average over batches), and returns it afterwards
            self.log("batch_idx", batch_idx)
            self.log("test_loss", loss, on_step=False, on_epoch=True)
            self.log("test_acc", acc, on_step=False, on_epoch=True)

    def test_step(  # type: ignore
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:  # type: ignore
        """Test step

        Args:
            - batch (Tuple[Tensor, Tensor]): Batch of the testing data.
            - batch_idx (int): Index of the batch.
        """
        imgs, labels = batch
        preds: Tensor = self.model(imgs)
        loss: Tensor = self.loss_module(preds, labels)
        preds = self.model(imgs).argmax(dim=-1)
        acc: Tensor = (labels == preds).float().mean()

        # Logs the fl-related metrics if fl_hparams is not None
        if self.fl_hparams is not None:
            self.log(
                f"{self.fl_hparams.get('experiment_name', 'default')}_test_loss",
                loss,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{self.fl_hparams.get('experiment_name', 'default')}_test_acc",
                acc,
                on_step=False,
                on_epoch=True,
            )
        else:
            # By default logs it per epoch (weighted average over batches), and returns it afterwards
            self.log("batch_idx", batch_idx)
            self.log("test_loss", loss, on_step=False, on_epoch=True)
            self.log("test_acc", acc, on_step=False, on_epoch=True)
