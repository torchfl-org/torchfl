#!/usr/bin/env python

"""Contains the PyTorch Lightning wrapper modules for CIFAR10 and CIFAR100 dataset."""

import enum
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor, optim

from torchfl.compatibility import OPTIMIZERS_BY_NAME, OPTIMIZERS_TYPE
from torchfl.federated.fl_params import FLParams
from torchfl.models.core.cifar.cifar10.alexnet import AlexNet as CIFAR10AlexNet
from torchfl.models.core.cifar.cifar10.densenet import (
    DenseNet121 as CIFAR10DenseNet121,
)
from torchfl.models.core.cifar.cifar10.densenet import (
    DenseNet161 as CIFAR10DenseNet161,
)
from torchfl.models.core.cifar.cifar10.densenet import (
    DenseNet169 as CIFAR10DenseNet169,
)
from torchfl.models.core.cifar.cifar10.densenet import (
    DenseNet201 as CIFAR10DenseNet201,
)
from torchfl.models.core.cifar.cifar10.lenet import LeNet as CIFAR10LeNet
from torchfl.models.core.cifar.cifar10.mobilenet import (
    MobileNetV2 as CIFAR10MobileNetV2,
)
from torchfl.models.core.cifar.cifar10.mobilenet import (
    MobileNetV3Large as CIFAR10MobileNetV3Large,
)
from torchfl.models.core.cifar.cifar10.mobilenet import (
    MobileNetV3Small as CIFAR10MobileNetV3Small,
)
from torchfl.models.core.cifar.cifar10.resnet import (
    ResNet18 as CIFAR10ResNet18,
)
from torchfl.models.core.cifar.cifar10.resnet import (
    ResNet34 as CIFAR10ResNet34,
)
from torchfl.models.core.cifar.cifar10.resnet import (
    ResNet50 as CIFAR10ResNet50,
)
from torchfl.models.core.cifar.cifar10.resnet import (
    ResNet101 as CIFAR10ResNet101,
)
from torchfl.models.core.cifar.cifar10.resnet import (
    ResNet152 as CIFAR10ResNet152,
)
from torchfl.models.core.cifar.cifar10.resnet import (
    ResNext50_32X4D as CIFAR10ResNext50_32X4D,
)
from torchfl.models.core.cifar.cifar10.resnet import (
    ResNext101_32X8D as CIFAR10ResNext101_32X8D,
)
from torchfl.models.core.cifar.cifar10.resnet import (
    WideResNet50_2 as CIFAR10WideResNet50_2,
)
from torchfl.models.core.cifar.cifar10.resnet import (
    WideResNet101_2 as CIFAR10WideResNet101_2,
)
from torchfl.models.core.cifar.cifar10.shufflenetv2 import (
    ShuffleNetv2_x0_5 as CIFAR10ShuffleNetv2_x0_5,
)
from torchfl.models.core.cifar.cifar10.shufflenetv2 import (
    ShuffleNetv2_x1_0 as CIFAR10ShuffleNetv2_x1_0,
)
from torchfl.models.core.cifar.cifar10.shufflenetv2 import (
    ShuffleNetv2_x1_5 as CIFAR10ShuffleNetv2_x1_5,
)
from torchfl.models.core.cifar.cifar10.shufflenetv2 import (
    ShuffleNetv2_x2_0 as CIFAR10ShuffleNetv2_x2_0,
)
from torchfl.models.core.cifar.cifar10.squeezenet import (
    SqueezeNet1_0 as CIFAR10SqueezeNet1_0,
)
from torchfl.models.core.cifar.cifar10.squeezenet import (
    SqueezeNet1_1 as CIFAR10SqueezeNet1_1,
)
from torchfl.models.core.cifar.cifar10.vgg import VGG11 as CIFAR10VGG11
from torchfl.models.core.cifar.cifar10.vgg import VGG11_BN as CIFAR10VGG11_BN
from torchfl.models.core.cifar.cifar10.vgg import VGG13 as CIFAR10VGG13
from torchfl.models.core.cifar.cifar10.vgg import VGG13_BN as CIFAR10VGG13_BN
from torchfl.models.core.cifar.cifar10.vgg import VGG16 as CIFAR10VGG16
from torchfl.models.core.cifar.cifar10.vgg import VGG16_BN as CIFAR10VGG16_BN
from torchfl.models.core.cifar.cifar10.vgg import VGG19 as CIFAR10VGG19
from torchfl.models.core.cifar.cifar10.vgg import VGG19_BN as CIFAR10VGG19_BN
from torchfl.models.core.cifar.cifar100.alexnet import (
    AlexNet as CIFAR100AlexNet,
)
from torchfl.models.core.cifar.cifar100.densenet import (
    DenseNet121 as CIFAR100DenseNet121,
)
from torchfl.models.core.cifar.cifar100.densenet import (
    DenseNet161 as CIFAR100DenseNet161,
)
from torchfl.models.core.cifar.cifar100.densenet import (
    DenseNet169 as CIFAR100DenseNet169,
)
from torchfl.models.core.cifar.cifar100.densenet import (
    DenseNet201 as CIFAR100DenseNet201,
)
from torchfl.models.core.cifar.cifar100.lenet import LeNet as CIFAR100LeNet
from torchfl.models.core.cifar.cifar100.mobilenet import (
    MobileNetV2 as CIFAR100MobileNetV2,
)
from torchfl.models.core.cifar.cifar100.mobilenet import (
    MobileNetV3Large as CIFAR100MobileNetV3Large,
)
from torchfl.models.core.cifar.cifar100.mobilenet import (
    MobileNetV3Small as CIFAR100MobileNetV3Small,
)
from torchfl.models.core.cifar.cifar100.resnet import (
    ResNet18 as CIFAR100ResNet18,
)
from torchfl.models.core.cifar.cifar100.resnet import (
    ResNet34 as CIFAR100ResNet34,
)
from torchfl.models.core.cifar.cifar100.resnet import (
    ResNet50 as CIFAR100ResNet50,
)
from torchfl.models.core.cifar.cifar100.resnet import (
    ResNet101 as CIFAR100ResNet101,
)
from torchfl.models.core.cifar.cifar100.resnet import (
    ResNet152 as CIFAR100ResNet152,
)
from torchfl.models.core.cifar.cifar100.resnet import (
    ResNext50_32X4D as CIFAR100ResNext50_32X4D,
)
from torchfl.models.core.cifar.cifar100.resnet import (
    ResNext101_32X8D as CIFAR100ResNext101_32X8D,
)
from torchfl.models.core.cifar.cifar100.resnet import (
    WideResNet50_2 as CIFAR100WideResNet50_2,
)
from torchfl.models.core.cifar.cifar100.resnet import (
    WideResNet101_2 as CIFAR100WideResNet101_2,
)
from torchfl.models.core.cifar.cifar100.shufflenetv2 import (
    ShuffleNetv2_x0_5 as CIFAR100ShuffleNetv2_x0_5,
)
from torchfl.models.core.cifar.cifar100.shufflenetv2 import (
    ShuffleNetv2_x1_0 as CIFAR100ShuffleNetv2_x1_0,
)
from torchfl.models.core.cifar.cifar100.shufflenetv2 import (
    ShuffleNetv2_x1_5 as CIFAR100ShuffleNetv2_x1_5,
)
from torchfl.models.core.cifar.cifar100.shufflenetv2 import (
    ShuffleNetv2_x2_0 as CIFAR100ShuffleNetv2_x2_0,
)
from torchfl.models.core.cifar.cifar100.squeezenet import (
    SqueezeNet1_0 as CIFAR100SqueezeNet1_0,
)
from torchfl.models.core.cifar.cifar100.squeezenet import (
    SqueezeNet1_1 as CIFAR100SqueezeNet1_1,
)
from torchfl.models.core.cifar.cifar100.vgg import VGG11 as CIFAR100VGG11
from torchfl.models.core.cifar.cifar100.vgg import VGG11_BN as CIFAR100VGG11_BN
from torchfl.models.core.cifar.cifar100.vgg import VGG13 as CIFAR100VGG13
from torchfl.models.core.cifar.cifar100.vgg import VGG13_BN as CIFAR100VGG13_BN
from torchfl.models.core.cifar.cifar100.vgg import VGG16 as CIFAR100VGG16
from torchfl.models.core.cifar.cifar100.vgg import VGG16_BN as CIFAR100VGG16_BN
from torchfl.models.core.cifar.cifar100.vgg import VGG19 as CIFAR100VGG19
from torchfl.models.core.cifar.cifar100.vgg import VGG19_BN as CIFAR100VGG19_BN

pl.seed_everything(42)

###############
# Begin Utils #
###############

CIFAR_MODELS: list[str] = [
    "alexnet",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "lenet",
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


class CIFAR_MODELS_ENUM(enum.Enum):
    """Enum for supported CIFAR models."""

    ALEXNET = "alexnet"
    DENSENET121 = "densenet121"
    DENSENET161 = "densenet161"
    DENSENET169 = "densenet169"
    DENSENET201 = "densenet201"
    LENET = "lenet"
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


CIFAR10_MODEL_TYPE = (
    type[CIFAR10AlexNet]
    | type[CIFAR10DenseNet121]
    | type[CIFAR10DenseNet161]
    | type[CIFAR10DenseNet169]
    | type[CIFAR10DenseNet201]
    | type[CIFAR10LeNet]
    | type[CIFAR10MobileNetV2]
    | type[CIFAR10MobileNetV3Large]
    | type[CIFAR10MobileNetV3Small]
    | type[CIFAR10ResNet18]
    | type[CIFAR10ResNet34]
    | type[CIFAR10ResNet50]
    | type[CIFAR10ResNet101]
    | type[CIFAR10ResNet152]
    | type[CIFAR10ResNext50_32X4D]
    | type[CIFAR10ResNext101_32X8D]
    | type[CIFAR10ShuffleNetv2_x0_5]
    | type[CIFAR10ShuffleNetv2_x1_0]
    | type[CIFAR10ShuffleNetv2_x1_5]
    | type[CIFAR10ShuffleNetv2_x2_0]
    | type[CIFAR10SqueezeNet1_0]
    | type[CIFAR10SqueezeNet1_1]
    | type[CIFAR10VGG11]
    | type[CIFAR10VGG11_BN]
    | type[CIFAR10VGG13]
    | type[CIFAR10VGG13_BN]
    | type[CIFAR10VGG16]
    | type[CIFAR10VGG16_BN]
    | type[CIFAR10VGG19]
    | type[CIFAR10VGG19_BN]
)

CIFAR100_MODEL_TYPE = (
    type[CIFAR100AlexNet]
    | type[CIFAR100DenseNet121]
    | type[CIFAR100DenseNet161]
    | type[CIFAR100DenseNet169]
    | type[CIFAR100DenseNet201]
    | type[CIFAR100LeNet]
    | type[CIFAR100MobileNetV2]
    | type[CIFAR100MobileNetV3Large]
    | type[CIFAR100MobileNetV3Small]
    | type[CIFAR100ResNet18]
    | type[CIFAR100ResNet34]
    | type[CIFAR100ResNet50]
    | type[CIFAR100ResNet101]
    | type[CIFAR100ResNet152]
    | type[CIFAR100ResNext50_32X4D]
    | type[CIFAR100ResNext101_32X8D]
    | type[CIFAR100ShuffleNetv2_x0_5]
    | type[CIFAR100ShuffleNetv2_x1_0]
    | type[CIFAR100ShuffleNetv2_x1_5]
    | type[CIFAR100ShuffleNetv2_x2_0]
    | type[CIFAR100SqueezeNet1_0]
    | type[CIFAR100SqueezeNet1_1]
    | type[CIFAR100VGG11]
    | type[CIFAR100VGG11_BN]
    | type[CIFAR100VGG13]
    | type[CIFAR100VGG13_BN]
    | type[CIFAR100VGG16]
    | type[CIFAR100VGG16_BN]
    | type[CIFAR100VGG19]
    | type[CIFAR100VGG19_BN]
)

CIFAR10_MODELS_MAPPING: dict[str, CIFAR10_MODEL_TYPE] = {
    "alexnet": CIFAR10AlexNet,
    "densenet121": CIFAR10DenseNet121,
    "densenet161": CIFAR10DenseNet161,
    "densenet169": CIFAR10DenseNet169,
    "densenet201": CIFAR10DenseNet201,
    "lenet": CIFAR10LeNet,
    "mobilenetv2": CIFAR10MobileNetV2,
    "mobilenetv3small": CIFAR10MobileNetV3Small,
    "mobilenetv3large": CIFAR10MobileNetV3Large,
    "resnet18": CIFAR10ResNet18,
    "resnet34": CIFAR10ResNet34,
    "resnet50": CIFAR10ResNet50,
    "resnet101": CIFAR10ResNet101,
    "resnet152": CIFAR10ResNet152,
    "resnext50_32x4d": CIFAR10ResNext50_32X4D,
    "resnext101_32x8d": CIFAR10ResNext101_32X8D,
    "wideresnet50_2": CIFAR10WideResNet50_2,
    "wideresnet101_2": CIFAR10WideResNet101_2,
    "shufflenetv2_x0_5": CIFAR10ShuffleNetv2_x0_5,
    "shufflenetv2_x1_0": CIFAR10ShuffleNetv2_x1_0,
    "shufflenetv2_x1_5": CIFAR10ShuffleNetv2_x1_5,
    "shufflenetv2_x2_0": CIFAR10ShuffleNetv2_x2_0,
    "squeezenet1_0": CIFAR10SqueezeNet1_0,
    "squeezenet1_1": CIFAR10SqueezeNet1_1,
    "vgg11": CIFAR10VGG11,
    "vgg11_bn": CIFAR10VGG11_BN,
    "vgg13": CIFAR10VGG13,
    "vgg13_bn": CIFAR10VGG13_BN,
    "vgg16": CIFAR10VGG16,
    "vgg16_bn": CIFAR10VGG16_BN,
    "vgg19": CIFAR10VGG19,
    "vgg19_bn": CIFAR10VGG19_BN,
}

CIFAR100_MODELS_MAPPING: dict[str, CIFAR100_MODEL_TYPE] = {
    "alexnet": CIFAR100AlexNet,
    "densenet121": CIFAR100DenseNet121,
    "densenet161": CIFAR100DenseNet161,
    "densenet169": CIFAR100DenseNet169,
    "densenet201": CIFAR100DenseNet201,
    "lenet": CIFAR100LeNet,
    "mobilenetv2": CIFAR100MobileNetV2,
    "mobilenetv3small": CIFAR100MobileNetV3Small,
    "mobilenetv3large": CIFAR100MobileNetV3Large,
    "resnet18": CIFAR100ResNet18,
    "resnet34": CIFAR100ResNet34,
    "resnet50": CIFAR100ResNet50,
    "resnet101": CIFAR100ResNet101,
    "resnet152": CIFAR100ResNet152,
    "resnext50_32x4d": CIFAR100ResNext50_32X4D,
    "resnext101_32x8d": CIFAR100ResNext101_32X8D,
    "wideresnet50_2": CIFAR100WideResNet50_2,
    "wideresnet101_2": CIFAR100WideResNet101_2,
    "shufflenetv2_x0_5": CIFAR100ShuffleNetv2_x0_5,
    "shufflenetv2_x1_0": CIFAR100ShuffleNetv2_x1_0,
    "shufflenetv2_x1_5": CIFAR100ShuffleNetv2_x1_5,
    "shufflenetv2_x2_0": CIFAR100ShuffleNetv2_x2_0,
    "squeezenet1_0": CIFAR100SqueezeNet1_0,
    "squeezenet1_1": CIFAR100SqueezeNet1_1,
    "vgg11": CIFAR100VGG11,
    "vgg11_bn": CIFAR100VGG11_BN,
    "vgg13": CIFAR100VGG13,
    "vgg13_bn": CIFAR100VGG13_BN,
    "vgg16": CIFAR100VGG16,
    "vgg16_bn": CIFAR100VGG16_BN,
    "vgg19": CIFAR100VGG19,
    "vgg19_bn": CIFAR100VGG19_BN,
}


def create_model(
    dataset_name: str,
    model_name: str,
    model_hparams: dict[str, CIFAR10_MODEL_TYPE | CIFAR100_MODEL_TYPE]
    | None = None,
) -> nn.Module:
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
    if dataset_name == "cifar10":
        if model_name not in CIFAR10_MODELS_MAPPING:
            raise ValueError(
                f"{model_name}: Invalid model name. Not supported for this dataset."
            )
        else:
            return (
                CIFAR10_MODELS_MAPPING[model_name](**model_hparams)
                if model_hparams
                else CIFAR10_MODELS_MAPPING[model_name]()
            )

    elif dataset_name == "cifar100":
        if model_name not in CIFAR100_MODELS_MAPPING:
            raise ValueError(
                f"{model_name}: Invalid model name. Not supported for this dataset."
            )
        else:
            return (
                CIFAR100_MODELS_MAPPING[model_name](**model_hparams)
                if model_hparams
                else CIFAR100_MODELS_MAPPING[model_name]()
            )

    else:
        raise ValueError(
            f"{dataset_name}: Invalid dataset name. Not a supported dataset."
        )


###############
# End Utils #
###############


class CIFAR10(pl.LightningModule):
    """PyTorch Lightning wrapper for CIFAR10 dataset."""

    def __init__(
        self,
        model_name: CIFAR_MODELS_ENUM,
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
                dataset_name="cifar10",
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


class CIFAR100(pl.LightningModule):
    """PyTorch Lightning wrapper for CIFAR100 dataset."""

    def __init__(
        self,
        model_name: CIFAR_MODELS_ENUM,
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
                dataset_name="cifar100",
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
