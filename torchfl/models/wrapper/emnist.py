#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the PyTorch Lightning wrapper modules for all EMNIST datasets."""

import enum
from typing import List, Optional, Type, Union, Dict, Any, Tuple
from torchfl.federated.fl_params import FLParams
from torchfl.models.core.emnist.balanced.alexnet import AlexNet as BalancedAlexNet
from torchfl.models.core.emnist.balanced.densenet import (
    DenseNet121 as BalancedDenseNet121,
    DenseNet161 as BalancedDenseNet161,
    DenseNet169 as BalancedDenseNet169,
    DenseNet201 as BalancedDenseNet201,
)
from torchfl.models.core.emnist.balanced.lenet import LeNet as BalancedLeNet
from torchfl.models.core.emnist.balanced.mlp import MLP as BalancedMLP
from torchfl.models.core.emnist.balanced.mobilenet import (
    MobileNetV2 as BalancedMobileNetV2,
    MobileNetV3Small as BalancedMobileNetV3Small,
    MobileNetV3Large as BalancedMobileNetV3Large,
)
from torchfl.models.core.emnist.balanced.resnet import (
    ResNet18 as BalancedResNet18,
    ResNet34 as BalancedResNet34,
    ResNet50 as BalancedResNet50,
    ResNet101 as BalancedResNet101,
    ResNet152 as BalancedResNet152,
    ResNext50_32X4D as BalancedResNext50_32X4D,
    ResNext101_32X8D as BalancedResNext101_32X8D,
    WideResNet50_2 as BalancedWideResNet50_2,
    WideResNet101_2 as BalancedWideResNet101_2,
)
from torchfl.models.core.emnist.balanced.shufflenetv2 import (
    ShuffleNetv2_x0_5 as BalancedShuffleNetv2_x0_5,
    ShuffleNetv2_x1_0 as BalancedShuffleNetv2_x1_0,
    ShuffleNetv2_x1_5 as BalancedShuffleNetv2_x1_5,
    ShuffleNetv2_x2_0 as BalancedShuffleNetv2_x2_0,
)
from torchfl.models.core.emnist.balanced.squeezenet import (
    SqueezeNet1_0 as BalancedSqueezeNet1_0,
    SqueezeNet1_1 as BalancedSqueezeNet1_1,
)
from torchfl.models.core.emnist.balanced.vgg import (
    VGG11 as BalancedVGG11,
    VGG11_BN as BalancedVGG11_BN,
    VGG13 as BalancedVGG13,
    VGG13_BN as BalancedVGG13_BN,
    VGG16 as BalancedVGG16,
    VGG16_BN as BalancedVGG16_BN,
    VGG19 as BalancedVGG19,
    VGG19_BN as BalancedVGG19_BN,
)
from torchfl.models.core.emnist.byclass.alexnet import AlexNet as ByClassAlexNet
from torchfl.models.core.emnist.byclass.densenet import (
    DenseNet121 as ByClassDenseNet121,
    DenseNet161 as ByClassDenseNet161,
    DenseNet169 as ByClassDenseNet169,
    DenseNet201 as ByClassDenseNet201,
)
from torchfl.models.core.emnist.byclass.lenet import LeNet as ByClassLeNet
from torchfl.models.core.emnist.byclass.mlp import MLP as ByClassMLP
from torchfl.models.core.emnist.byclass.mobilenet import (
    MobileNetV2 as ByClassMobileNetV2,
    MobileNetV3Small as ByClassMobileNetV3Small,
    MobileNetV3Large as ByClassMobileNetV3Large,
)
from torchfl.models.core.emnist.byclass.resnet import (
    ResNet18 as ByClassResNet18,
    ResNet34 as ByClassResNet34,
    ResNet50 as ByClassResNet50,
    ResNet101 as ByClassResNet101,
    ResNet152 as ByClassResNet152,
    ResNext50_32X4D as ByClassResNext50_32X4D,
    ResNext101_32X8D as ByClassResNext101_32X8D,
    WideResNet50_2 as ByClassWideResNet50_2,
    WideResNet101_2 as ByClassWideResNet101_2,
)
from torchfl.models.core.emnist.byclass.shufflenetv2 import (
    ShuffleNetv2_x0_5 as ByClassShuffleNetv2_x0_5,
    ShuffleNetv2_x1_0 as ByClassShuffleNetv2_x1_0,
    ShuffleNetv2_x1_5 as ByClassShuffleNetv2_x1_5,
    ShuffleNetv2_x2_0 as ByClassShuffleNetv2_x2_0,
)
from torchfl.models.core.emnist.byclass.squeezenet import (
    SqueezeNet1_0 as ByClassSqueezeNet1_0,
    SqueezeNet1_1 as ByClassSqueezeNet1_1,
)
from torchfl.models.core.emnist.byclass.vgg import (
    VGG11 as ByClassVGG11,
    VGG11_BN as ByClassVGG11_BN,
    VGG13 as ByClassVGG13,
    VGG13_BN as ByClassVGG13_BN,
    VGG16 as ByClassVGG16,
    VGG16_BN as ByClassVGG16_BN,
    VGG19 as ByClassVGG19,
    VGG19_BN as ByClassVGG19_BN,
)
from torchfl.models.core.emnist.bymerge.alexnet import AlexNet as ByMergeAlexNet
from torchfl.models.core.emnist.bymerge.densenet import (
    DenseNet121 as ByMergeDenseNet121,
    DenseNet161 as ByMergeDenseNet161,
    DenseNet169 as ByMergeDenseNet169,
    DenseNet201 as ByMergeDenseNet201,
)
from torchfl.models.core.emnist.bymerge.lenet import LeNet as ByMergeLeNet
from torchfl.models.core.emnist.bymerge.mlp import MLP as ByMergeMLP
from torchfl.models.core.emnist.bymerge.mobilenet import (
    MobileNetV2 as ByMergeMobileNetV2,
    MobileNetV3Small as ByMergeMobileNetV3Small,
    MobileNetV3Large as ByMergeMobileNetV3Large,
)
from torchfl.models.core.emnist.bymerge.resnet import (
    ResNet18 as ByMergeResNet18,
    ResNet34 as ByMergeResNet34,
    ResNet50 as ByMergeResNet50,
    ResNet101 as ByMergeResNet101,
    ResNet152 as ByMergeResNet152,
    ResNext50_32X4D as ByMergeResNext50_32X4D,
    ResNext101_32X8D as ByMergeResNext101_32X8D,
    WideResNet50_2 as ByMergeWideResNet50_2,
    WideResNet101_2 as ByMergeWideResNet101_2,
)
from torchfl.models.core.emnist.bymerge.shufflenetv2 import (
    ShuffleNetv2_x0_5 as ByMergeShuffleNetv2_x0_5,
    ShuffleNetv2_x1_0 as ByMergeShuffleNetv2_x1_0,
    ShuffleNetv2_x1_5 as ByMergeShuffleNetv2_x1_5,
    ShuffleNetv2_x2_0 as ByMergeShuffleNetv2_x2_0,
)
from torchfl.models.core.emnist.bymerge.squeezenet import (
    SqueezeNet1_0 as ByMergeSqueezeNet1_0,
    SqueezeNet1_1 as ByMergeSqueezeNet1_1,
)
from torchfl.models.core.emnist.bymerge.vgg import (
    VGG11 as ByMergeVGG11,
    VGG11_BN as ByMergeVGG11_BN,
    VGG13 as ByMergeVGG13,
    VGG13_BN as ByMergeVGG13_BN,
    VGG16 as ByMergeVGG16,
    VGG16_BN as ByMergeVGG16_BN,
    VGG19 as ByMergeVGG19,
    VGG19_BN as ByMergeVGG19_BN,
)
from torchfl.models.core.emnist.digits.alexnet import AlexNet as DigitsAlexNet
from torchfl.models.core.emnist.digits.densenet import (
    DenseNet121 as DigitsDenseNet121,
    DenseNet161 as DigitsDenseNet161,
    DenseNet169 as DigitsDenseNet169,
    DenseNet201 as DigitsDenseNet201,
)
from torchfl.models.core.emnist.digits.lenet import LeNet as DigitsLeNet
from torchfl.models.core.emnist.digits.mlp import MLP as DigitsMLP
from torchfl.models.core.emnist.digits.mobilenet import (
    MobileNetV2 as DigitsMobileNetV2,
    MobileNetV3Small as DigitsMobileNetV3Small,
    MobileNetV3Large as DigitsMobileNetV3Large,
)
from torchfl.models.core.emnist.digits.resnet import (
    ResNet18 as DigitsResNet18,
    ResNet34 as DigitsResNet34,
    ResNet50 as DigitsResNet50,
    ResNet101 as DigitsResNet101,
    ResNet152 as DigitsResNet152,
    ResNext50_32X4D as DigitsResNext50_32X4D,
    ResNext101_32X8D as DigitsResNext101_32X8D,
    WideResNet50_2 as DigitsWideResNet50_2,
    WideResNet101_2 as DigitsWideResNet101_2,
)
from torchfl.models.core.emnist.digits.shufflenetv2 import (
    ShuffleNetv2_x0_5 as DigitsShuffleNetv2_x0_5,
    ShuffleNetv2_x1_0 as DigitsShuffleNetv2_x1_0,
    ShuffleNetv2_x1_5 as DigitsShuffleNetv2_x1_5,
    ShuffleNetv2_x2_0 as DigitsShuffleNetv2_x2_0,
)
from torchfl.models.core.emnist.digits.squeezenet import (
    SqueezeNet1_0 as DigitsSqueezeNet1_0,
    SqueezeNet1_1 as DigitsSqueezeNet1_1,
)
from torchfl.models.core.emnist.digits.vgg import (
    VGG11 as DigitsVGG11,
    VGG11_BN as DigitsVGG11_BN,
    VGG13 as DigitsVGG13,
    VGG13_BN as DigitsVGG13_BN,
    VGG16 as DigitsVGG16,
    VGG16_BN as DigitsVGG16_BN,
    VGG19 as DigitsVGG19,
    VGG19_BN as DigitsVGG19_BN,
)
from torchfl.models.core.emnist.letters.alexnet import AlexNet as LettersAlexNet
from torchfl.models.core.emnist.letters.densenet import (
    DenseNet121 as LettersDenseNet121,
    DenseNet161 as LettersDenseNet161,
    DenseNet169 as LettersDenseNet169,
    DenseNet201 as LettersDenseNet201,
)
from torchfl.models.core.emnist.letters.lenet import LeNet as LettersLeNet
from torchfl.models.core.emnist.letters.mlp import MLP as LettersMLP
from torchfl.models.core.emnist.letters.mobilenet import (
    MobileNetV2 as LettersMobileNetV2,
    MobileNetV3Small as LettersMobileNetV3Small,
    MobileNetV3Large as LettersMobileNetV3Large,
)
from torchfl.models.core.emnist.letters.resnet import (
    ResNet18 as LettersResNet18,
    ResNet34 as LettersResNet34,
    ResNet50 as LettersResNet50,
    ResNet101 as LettersResNet101,
    ResNet152 as LettersResNet152,
    ResNext50_32X4D as LettersResNext50_32X4D,
    ResNext101_32X8D as LettersResNext101_32X8D,
    WideResNet50_2 as LettersWideResNet50_2,
    WideResNet101_2 as LettersWideResNet101_2,
)
from torchfl.models.core.emnist.letters.shufflenetv2 import (
    ShuffleNetv2_x0_5 as LettersShuffleNetv2_x0_5,
    ShuffleNetv2_x1_0 as LettersShuffleNetv2_x1_0,
    ShuffleNetv2_x1_5 as LettersShuffleNetv2_x1_5,
    ShuffleNetv2_x2_0 as LettersShuffleNetv2_x2_0,
)
from torchfl.models.core.emnist.letters.squeezenet import (
    SqueezeNet1_0 as LettersSqueezeNet1_0,
    SqueezeNet1_1 as LettersSqueezeNet1_1,
)
from torchfl.models.core.emnist.letters.vgg import (
    VGG11 as LettersVGG11,
    VGG11_BN as LettersVGG11_BN,
    VGG13 as LettersVGG13,
    VGG13_BN as LettersVGG13_BN,
    VGG16 as LettersVGG16,
    VGG16_BN as LettersVGG16_BN,
    VGG19 as LettersVGG19,
    VGG19_BN as LettersVGG19_BN,
)
from torchfl.models.core.emnist.mnist.alexnet import AlexNet as MNISTAlexNet
from torchfl.models.core.emnist.mnist.densenet import (
    DenseNet121 as MNISTDenseNet121,
    DenseNet161 as MNISTDenseNet161,
    DenseNet169 as MNISTDenseNet169,
    DenseNet201 as MNISTDenseNet201,
)
from torchfl.models.core.emnist.mnist.lenet import LeNet as MNISTLeNet
from torchfl.models.core.emnist.mnist.mlp import MLP as MNISTMLP
from torchfl.models.core.emnist.mnist.mobilenet import (
    MobileNetV2 as MNISTMobileNetV2,
    MobileNetV3Small as MNISTMobileNetV3Small,
    MobileNetV3Large as MNISTMobileNetV3Large,
)
from torchfl.models.core.emnist.mnist.resnet import (
    ResNet18 as MNISTResNet18,
    ResNet34 as MNISTResNet34,
    ResNet50 as MNISTResNet50,
    ResNet101 as MNISTResNet101,
    ResNet152 as MNISTResNet152,
    ResNext50_32X4D as MNISTResNext50_32X4D,
    ResNext101_32X8D as MNISTResNext101_32X8D,
    WideResNet50_2 as MNISTWideResNet50_2,
    WideResNet101_2 as MNISTWideResNet101_2,
)
from torchfl.models.core.emnist.mnist.shufflenetv2 import (
    ShuffleNetv2_x0_5 as MNISTShuffleNetv2_x0_5,
    ShuffleNetv2_x1_0 as MNISTShuffleNetv2_x1_0,
    ShuffleNetv2_x1_5 as MNISTShuffleNetv2_x1_5,
    ShuffleNetv2_x2_0 as MNISTShuffleNetv2_x2_0,
)
from torchfl.models.core.emnist.mnist.squeezenet import (
    SqueezeNet1_0 as MNISTSqueezeNet1_0,
    SqueezeNet1_1 as MNISTSqueezeNet1_1,
)
from torchfl.models.core.emnist.mnist.vgg import (
    VGG11 as MNISTVGG11,
    VGG11_BN as MNISTVGG11_BN,
    VGG13 as MNISTVGG13,
    VGG13_BN as MNISTVGG13_BN,
    VGG16 as MNISTVGG16,
    VGG16_BN as MNISTVGG16_BN,
    VGG19 as MNISTVGG19,
    VGG19_BN as MNISTVGG19_BN,
)
import pytorch_lightning as pl
import torch.nn as nn
from torchfl.compatibility import OPTIMIZERS_TYPE, OPTIMIZERS_BY_NAME
from torch import Tensor, optim

pl.seed_everything(42)

###############
# Begin Utils #
###############
EMNIST_MODELS: List[str] = [
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


class EMNIST_MODELS_ENUM(enum.Enum):
    """Enum for supported EMNIST models."""

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


EMNIST_BALANCED_MODEL_TYPE = Union[
    Type[BalancedAlexNet],
    Type[BalancedDenseNet121],
    Type[BalancedDenseNet161],
    Type[BalancedDenseNet169],
    Type[BalancedDenseNet201],
    Type[BalancedLeNet],
    Type[BalancedMLP],
    Type[BalancedMobileNetV2],
    Type[BalancedMobileNetV3Large],
    Type[BalancedMobileNetV3Small],
    Type[BalancedResNet18],
    Type[BalancedResNet34],
    Type[BalancedResNet50],
    Type[BalancedResNet101],
    Type[BalancedResNet152],
    Type[BalancedResNext50_32X4D],
    Type[BalancedResNext101_32X8D],
    Type[BalancedWideResNet50_2],
    Type[BalancedWideResNet101_2],
    Type[BalancedShuffleNetv2_x0_5],
    Type[BalancedShuffleNetv2_x1_0],
    Type[BalancedShuffleNetv2_x1_5],
    Type[BalancedShuffleNetv2_x2_0],
    Type[BalancedSqueezeNet1_0],
    Type[BalancedSqueezeNet1_1],
    Type[BalancedVGG11],
    Type[BalancedVGG11_BN],
    Type[BalancedVGG13],
    Type[BalancedVGG13_BN],
    Type[BalancedVGG16],
    Type[BalancedVGG16_BN],
    Type[BalancedVGG19],
    Type[BalancedVGG19_BN],
]
EMNIST_BYCLASS_MODEL_TYPE = Union[
    Type[ByClassAlexNet],
    Type[ByClassDenseNet121],
    Type[ByClassDenseNet161],
    Type[ByClassDenseNet169],
    Type[ByClassDenseNet201],
    Type[ByClassLeNet],
    Type[ByClassMLP],
    Type[ByClassMobileNetV2],
    Type[ByClassMobileNetV3Large],
    Type[ByClassMobileNetV3Small],
    Type[ByClassResNet18],
    Type[ByClassResNet34],
    Type[ByClassResNet50],
    Type[ByClassResNet101],
    Type[ByClassResNet152],
    Type[ByClassResNext50_32X4D],
    Type[ByClassResNext101_32X8D],
    Type[ByClassWideResNet50_2],
    Type[ByClassWideResNet101_2],
    Type[ByClassShuffleNetv2_x0_5],
    Type[ByClassShuffleNetv2_x1_0],
    Type[ByClassShuffleNetv2_x1_5],
    Type[ByClassShuffleNetv2_x2_0],
    Type[ByClassSqueezeNet1_0],
    Type[ByClassSqueezeNet1_1],
    Type[ByClassVGG11],
    Type[ByClassVGG11_BN],
    Type[ByClassVGG13],
    Type[ByClassVGG13_BN],
    Type[ByClassVGG16],
    Type[ByClassVGG16_BN],
    Type[ByClassVGG19],
    Type[ByClassVGG19_BN],
]

EMNIST_BYMERGE_MODEL_TYPE = Union[
    Type[ByMergeAlexNet],
    Type[ByMergeDenseNet121],
    Type[ByMergeDenseNet161],
    Type[ByMergeDenseNet169],
    Type[ByMergeDenseNet201],
    Type[ByMergeLeNet],
    Type[ByMergeMLP],
    Type[ByMergeMobileNetV2],
    Type[ByMergeMobileNetV3Large],
    Type[ByMergeMobileNetV3Small],
    Type[ByMergeResNet18],
    Type[ByMergeResNet34],
    Type[ByMergeResNet50],
    Type[ByMergeResNet101],
    Type[ByMergeResNet152],
    Type[ByMergeResNext50_32X4D],
    Type[ByMergeResNext101_32X8D],
    Type[ByMergeWideResNet50_2],
    Type[ByMergeWideResNet101_2],
    Type[ByMergeShuffleNetv2_x0_5],
    Type[ByMergeShuffleNetv2_x1_0],
    Type[ByMergeShuffleNetv2_x1_5],
    Type[ByMergeShuffleNetv2_x2_0],
    Type[ByMergeSqueezeNet1_0],
    Type[ByMergeSqueezeNet1_1],
    Type[ByMergeVGG11],
    Type[ByMergeVGG11_BN],
    Type[ByMergeVGG13],
    Type[ByMergeVGG13_BN],
    Type[ByMergeVGG16],
    Type[ByMergeVGG16_BN],
    Type[ByMergeVGG19],
    Type[ByMergeVGG19_BN],
]

EMNIST_DIGITS_MODEL_TYPE = Union[
    Type[DigitsAlexNet],
    Type[DigitsDenseNet121],
    Type[DigitsDenseNet161],
    Type[DigitsDenseNet169],
    Type[DigitsDenseNet201],
    Type[DigitsLeNet],
    Type[DigitsMLP],
    Type[DigitsMobileNetV2],
    Type[DigitsMobileNetV3Large],
    Type[DigitsMobileNetV3Small],
    Type[DigitsResNet18],
    Type[DigitsResNet34],
    Type[DigitsResNet50],
    Type[DigitsResNet101],
    Type[DigitsResNet152],
    Type[DigitsResNext50_32X4D],
    Type[DigitsResNext101_32X8D],
    Type[DigitsWideResNet50_2],
    Type[DigitsWideResNet101_2],
    Type[DigitsShuffleNetv2_x0_5],
    Type[DigitsShuffleNetv2_x1_0],
    Type[DigitsShuffleNetv2_x1_5],
    Type[DigitsShuffleNetv2_x2_0],
    Type[DigitsSqueezeNet1_0],
    Type[DigitsSqueezeNet1_1],
    Type[DigitsVGG11],
    Type[DigitsVGG11_BN],
    Type[DigitsVGG13],
    Type[DigitsVGG13_BN],
    Type[DigitsVGG16],
    Type[DigitsVGG16_BN],
    Type[DigitsVGG19],
    Type[DigitsVGG19_BN],
]

EMNIST_LETTERS_MODEL_TYPE = Union[
    Type[LettersAlexNet],
    Type[LettersDenseNet121],
    Type[LettersDenseNet161],
    Type[LettersDenseNet169],
    Type[LettersDenseNet201],
    Type[LettersLeNet],
    Type[LettersMLP],
    Type[LettersMobileNetV2],
    Type[LettersMobileNetV3Large],
    Type[LettersMobileNetV3Small],
    Type[LettersResNet18],
    Type[LettersResNet34],
    Type[LettersResNet50],
    Type[LettersResNet101],
    Type[LettersResNet152],
    Type[LettersResNext50_32X4D],
    Type[LettersResNext101_32X8D],
    Type[LettersWideResNet50_2],
    Type[LettersWideResNet101_2],
    Type[LettersShuffleNetv2_x0_5],
    Type[LettersShuffleNetv2_x1_0],
    Type[LettersShuffleNetv2_x1_5],
    Type[LettersShuffleNetv2_x2_0],
    Type[LettersSqueezeNet1_0],
    Type[LettersSqueezeNet1_1],
    Type[LettersVGG11],
    Type[LettersVGG11_BN],
    Type[LettersVGG13],
    Type[LettersVGG13_BN],
    Type[LettersVGG16],
    Type[LettersVGG16_BN],
    Type[LettersVGG19],
    Type[LettersVGG19_BN],
]

EMNIST_MNIST_MODEL_TYPE = Union[
    Type[MNISTAlexNet],
    Type[MNISTDenseNet121],
    Type[MNISTDenseNet161],
    Type[MNISTDenseNet169],
    Type[MNISTDenseNet201],
    Type[MNISTLeNet],
    Type[MNISTMLP],
    Type[MNISTMobileNetV2],
    Type[MNISTMobileNetV3Large],
    Type[MNISTMobileNetV3Small],
    Type[MNISTResNet18],
    Type[MNISTResNet34],
    Type[MNISTResNet50],
    Type[MNISTResNet101],
    Type[MNISTResNet152],
    Type[MNISTResNext50_32X4D],
    Type[MNISTResNext101_32X8D],
    Type[MNISTWideResNet50_2],
    Type[MNISTWideResNet101_2],
    Type[MNISTShuffleNetv2_x0_5],
    Type[MNISTShuffleNetv2_x1_0],
    Type[MNISTShuffleNetv2_x1_5],
    Type[MNISTShuffleNetv2_x2_0],
    Type[MNISTSqueezeNet1_0],
    Type[MNISTSqueezeNet1_1],
    Type[MNISTVGG11],
    Type[MNISTVGG11_BN],
    Type[MNISTVGG13],
    Type[MNISTVGG13_BN],
    Type[MNISTVGG16],
    Type[MNISTVGG16_BN],
    Type[MNISTVGG19],
    Type[MNISTVGG19_BN],
]

EMNIST_BALANCED_MODELS_MAPPING: Dict[str, EMNIST_BALANCED_MODEL_TYPE] = {
    "alexnet": BalancedAlexNet,
    "densenet121": BalancedDenseNet121,
    "densenet161": BalancedDenseNet161,
    "densenet169": BalancedDenseNet169,
    "densenet201": BalancedDenseNet201,
    "lenet": BalancedLeNet,
    "mlp": BalancedMLP,
    "mobilenetv2": BalancedMobileNetV2,
    "mobilenetv3small": BalancedMobileNetV3Small,
    "mobilenetv3large": BalancedMobileNetV3Large,
    "resnet18": BalancedResNet18,
    "resnet34": BalancedResNet34,
    "resnet50": BalancedResNet50,
    "resnet101": BalancedResNet101,
    "resnet152": BalancedResNet152,
    "resnext50_32x4d": BalancedResNext50_32X4D,
    "resnext101_32x8d": BalancedResNext101_32X8D,
    "wideresnet50_2": BalancedWideResNet50_2,
    "wideresnet101_2": BalancedWideResNet101_2,
    "shufflenetv2_x0_5": BalancedShuffleNetv2_x0_5,
    "shufflenetv2_x1_0": BalancedShuffleNetv2_x1_0,
    "shufflenetv2_x1_5": BalancedShuffleNetv2_x1_5,
    "shufflenetv2_x2_0": BalancedShuffleNetv2_x2_0,
    "squeezenet1_0": BalancedSqueezeNet1_0,
    "squeezenet1_1": BalancedSqueezeNet1_1,
    "vgg11": BalancedVGG11,
    "vgg11_bn": BalancedVGG11_BN,
    "vgg13": BalancedVGG13,
    "vgg13_bn": BalancedVGG13_BN,
    "vgg16": BalancedVGG16,
    "vgg16_bn": BalancedVGG16_BN,
    "vgg19": BalancedVGG19,
    "vgg19_bn": BalancedVGG19_BN,
}

EMNIST_BYCLASS_MODELS_MAPPING: Dict[str, EMNIST_BYCLASS_MODEL_TYPE] = {
    "alexnet": ByClassAlexNet,
    "densenet121": ByClassDenseNet121,
    "densenet161": ByClassDenseNet161,
    "densenet169": ByClassDenseNet169,
    "densenet201": ByClassDenseNet201,
    "lenet": ByClassLeNet,
    "mlp": ByClassMLP,
    "mobilenetv2": ByClassMobileNetV2,
    "mobilenetv3small": ByClassMobileNetV3Small,
    "mobilenetv3large": ByClassMobileNetV3Large,
    "resnet18": ByClassResNet18,
    "resnet34": ByClassResNet34,
    "resnet50": ByClassResNet50,
    "resnet101": ByClassResNet101,
    "resnet152": ByClassResNet152,
    "resnext50_32x4d": ByClassResNext50_32X4D,
    "resnext101_32x8d": ByClassResNext101_32X8D,
    "wideresnet50_2": ByClassWideResNet50_2,
    "wideresnet101_2": ByClassWideResNet101_2,
    "shufflenetv2_x0_5": ByClassShuffleNetv2_x0_5,
    "shufflenetv2_x1_0": ByClassShuffleNetv2_x1_0,
    "shufflenetv2_x1_5": ByClassShuffleNetv2_x1_5,
    "shufflenetv2_x2_0": ByClassShuffleNetv2_x2_0,
    "squeezenet1_0": ByClassSqueezeNet1_0,
    "squeezenet1_1": ByClassSqueezeNet1_1,
    "vgg11": ByClassVGG11,
    "vgg11_bn": ByClassVGG11_BN,
    "vgg13": ByClassVGG13,
    "vgg13_bn": ByClassVGG13_BN,
    "vgg16": ByClassVGG16,
    "vgg16_bn": ByClassVGG16_BN,
    "vgg19": ByClassVGG19,
    "vgg19_bn": ByClassVGG19_BN,
}

EMNIST_BYMERGE_MODELS_MAPPING: Dict[str, EMNIST_BYMERGE_MODEL_TYPE] = {
    "alexnet": ByMergeAlexNet,
    "densenet121": ByMergeDenseNet121,
    "densenet161": ByMergeDenseNet161,
    "densenet169": ByMergeDenseNet169,
    "densenet201": ByMergeDenseNet201,
    "lenet": ByMergeLeNet,
    "mlp": ByMergeMLP,
    "mobilenetv2": ByMergeMobileNetV2,
    "mobilenetv3small": ByMergeMobileNetV3Small,
    "mobilenetv3large": ByMergeMobileNetV3Large,
    "resnet18": ByMergeResNet18,
    "resnet34": ByMergeResNet34,
    "resnet50": ByMergeResNet50,
    "resnet101": ByMergeResNet101,
    "resnet152": ByMergeResNet152,
    "resnext50_32x4d": ByMergeResNext50_32X4D,
    "resnext101_32x8d": ByMergeResNext101_32X8D,
    "wideresnet50_2": ByMergeWideResNet50_2,
    "wideresnet101_2": ByMergeWideResNet101_2,
    "shufflenetv2_x0_5": ByMergeShuffleNetv2_x0_5,
    "shufflenetv2_x1_0": ByMergeShuffleNetv2_x1_0,
    "shufflenetv2_x1_5": ByMergeShuffleNetv2_x1_5,
    "shufflenetv2_x2_0": ByMergeShuffleNetv2_x2_0,
    "squeezenet1_0": ByMergeSqueezeNet1_0,
    "squeezenet1_1": ByMergeSqueezeNet1_1,
    "vgg11": ByMergeVGG11,
    "vgg11_bn": ByMergeVGG11_BN,
    "vgg13": ByMergeVGG13,
    "vgg13_bn": ByMergeVGG13_BN,
    "vgg16": ByMergeVGG16,
    "vgg16_bn": ByMergeVGG16_BN,
    "vgg19": ByMergeVGG19,
    "vgg19_bn": ByMergeVGG19_BN,
}

EMNIST_DIGITS_MODELS_MAPPING: Dict[str, EMNIST_DIGITS_MODEL_TYPE] = {
    "alexnet": DigitsAlexNet,
    "densenet121": DigitsDenseNet121,
    "densenet161": DigitsDenseNet161,
    "densenet169": DigitsDenseNet169,
    "densenet201": DigitsDenseNet201,
    "lenet": DigitsLeNet,
    "mlp": DigitsMLP,
    "mobilenetv2": DigitsMobileNetV2,
    "mobilenetv3small": DigitsMobileNetV3Small,
    "mobilenetv3large": DigitsMobileNetV3Large,
    "resnet18": DigitsResNet18,
    "resnet34": DigitsResNet34,
    "resnet50": DigitsResNet50,
    "resnet101": DigitsResNet101,
    "resnet152": DigitsResNet152,
    "resnext50_32x4d": DigitsResNext50_32X4D,
    "resnext101_32x8d": DigitsResNext101_32X8D,
    "wideresnet50_2": DigitsWideResNet50_2,
    "wideresnet101_2": DigitsWideResNet101_2,
    "shufflenetv2_x0_5": DigitsShuffleNetv2_x0_5,
    "shufflenetv2_x1_0": DigitsShuffleNetv2_x1_0,
    "shufflenetv2_x1_5": DigitsShuffleNetv2_x1_5,
    "shufflenetv2_x2_0": DigitsShuffleNetv2_x2_0,
    "squeezenet1_0": DigitsSqueezeNet1_0,
    "squeezenet1_1": DigitsSqueezeNet1_1,
    "vgg11": DigitsVGG11,
    "vgg11_bn": DigitsVGG11_BN,
    "vgg13": DigitsVGG13,
    "vgg13_bn": DigitsVGG13_BN,
    "vgg16": DigitsVGG16,
    "vgg16_bn": DigitsVGG16_BN,
    "vgg19": DigitsVGG19,
    "vgg19_bn": DigitsVGG19_BN,
}

EMNIST_LETTERS_MODELS_MAPPING: Dict[str, EMNIST_LETTERS_MODEL_TYPE] = {
    "alexnet": LettersAlexNet,
    "densenet121": LettersDenseNet121,
    "densenet161": LettersDenseNet161,
    "densenet169": LettersDenseNet169,
    "densenet201": LettersDenseNet201,
    "lenet": LettersLeNet,
    "mlp": LettersMLP,
    "mobilenetv2": LettersMobileNetV2,
    "mobilenetv3small": LettersMobileNetV3Small,
    "mobilenetv3large": LettersMobileNetV3Large,
    "resnet18": LettersResNet18,
    "resnet34": LettersResNet34,
    "resnet50": LettersResNet50,
    "resnet101": LettersResNet101,
    "resnet152": LettersResNet152,
    "resnext50_32x4d": LettersResNext50_32X4D,
    "resnext101_32x8d": LettersResNext101_32X8D,
    "wideresnet50_2": LettersWideResNet50_2,
    "wideresnet101_2": LettersWideResNet101_2,
    "shufflenetv2_x0_5": LettersShuffleNetv2_x0_5,
    "shufflenetv2_x1_0": LettersShuffleNetv2_x1_0,
    "shufflenetv2_x1_5": LettersShuffleNetv2_x1_5,
    "shufflenetv2_x2_0": LettersShuffleNetv2_x2_0,
    "squeezenet1_0": LettersSqueezeNet1_0,
    "squeezenet1_1": LettersSqueezeNet1_1,
    "vgg11": LettersVGG11,
    "vgg11_bn": LettersVGG11_BN,
    "vgg13": LettersVGG13,
    "vgg13_bn": LettersVGG13_BN,
    "vgg16": LettersVGG16,
    "vgg16_bn": LettersVGG16_BN,
    "vgg19": LettersVGG19,
    "vgg19_bn": LettersVGG19_BN,
}

EMNIST_MNIST_MODELS_MAPPING: Dict[str, EMNIST_MNIST_MODEL_TYPE] = {
    "alexnet": MNISTAlexNet,
    "densenet121": MNISTDenseNet121,
    "densenet161": MNISTDenseNet161,
    "densenet169": MNISTDenseNet169,
    "densenet201": MNISTDenseNet201,
    "lenet": MNISTLeNet,
    "mlp": MNISTMLP,
    "mobilenetv2": MNISTMobileNetV2,
    "mobilenetv3small": MNISTMobileNetV3Small,
    "mobilenetv3large": MNISTMobileNetV3Large,
    "resnet18": MNISTResNet18,
    "resnet34": MNISTResNet34,
    "resnet50": MNISTResNet50,
    "resnet101": MNISTResNet101,
    "resnet152": MNISTResNet152,
    "resnext50_32x4d": MNISTResNext50_32X4D,
    "resnext101_32x8d": MNISTResNext101_32X8D,
    "wideresnet50_2": MNISTWideResNet50_2,
    "wideresnet101_2": MNISTWideResNet101_2,
    "shufflenetv2_x0_5": MNISTShuffleNetv2_x0_5,
    "shufflenetv2_x1_0": MNISTShuffleNetv2_x1_0,
    "shufflenetv2_x1_5": MNISTShuffleNetv2_x1_5,
    "shufflenetv2_x2_0": MNISTShuffleNetv2_x2_0,
    "squeezenet1_0": MNISTSqueezeNet1_0,
    "squeezenet1_1": MNISTSqueezeNet1_1,
    "vgg11": MNISTVGG11,
    "vgg11_bn": MNISTVGG11_BN,
    "vgg13": MNISTVGG13,
    "vgg13_bn": MNISTVGG13_BN,
    "vgg16": MNISTVGG16,
    "vgg16_bn": MNISTVGG16_BN,
    "vgg19": MNISTVGG19,
    "vgg19_bn": MNISTVGG19_BN,
}


def create_model(
    dataset_name: str, model_name: str, model_hparams: Optional[Dict[str, Any]] = None
) -> Union[
    EMNIST_BALANCED_MODEL_TYPE,
    EMNIST_BYCLASS_MODEL_TYPE,
    EMNIST_BYMERGE_MODEL_TYPE,
    EMNIST_DIGITS_MODEL_TYPE,
    EMNIST_LETTERS_MODEL_TYPE,
    EMNIST_MNIST_MODEL_TYPE,
]:
    """Helper function to create a model from the available options.

    Args:
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model for the dataset.
        model_hparams (Optional[Dict[str, Any]]): Hyperparameters for the model. Defaults to None.

    Returns:
        nn.Module: PyTorch model definition.

    Raises:
        ValueError: Unsupported dataset name or model.
    """
    if dataset_name == "balanced":
        if model_name not in EMNIST_BALANCED_MODELS_MAPPING:
            raise ValueError(
                f"{model_name}: Invalid model name. Not supported for this dataset."
            )
        else:
            if not model_hparams:
                return EMNIST_BALANCED_MODELS_MAPPING[model_name]()
            else:
                return EMNIST_BALANCED_MODELS_MAPPING[model_name](**model_hparams)
    elif dataset_name == "byclass":
        if model_name not in EMNIST_BYCLASS_MODELS_MAPPING:
            raise ValueError(
                f"{model_name}: Invalid model name. Not supported for this dataset."
            )
        else:
            if not model_hparams:
                return EMNIST_BYCLASS_MODELS_MAPPING[model_name]()
            else:
                return EMNIST_BYCLASS_MODELS_MAPPING[model_name](**model_hparams)
    elif dataset_name == "bymerge":
        if model_name not in EMNIST_BYMERGE_MODELS_MAPPING:
            raise ValueError(
                f"{model_name}: Invalid model name. Not supported for this dataset."
            )
        else:
            if not model_hparams:
                return EMNIST_BYMERGE_MODELS_MAPPING[model_name]()
            else:
                return EMNIST_BYMERGE_MODELS_MAPPING[model_name](**model_hparams)
    elif dataset_name == "letters":
        if model_name not in EMNIST_LETTERS_MODELS_MAPPING:
            raise ValueError(
                f"{model_name}: Invalid model name. Not supported for this dataset."
            )
        else:
            if not model_hparams:
                return EMNIST_LETTERS_MODELS_MAPPING[model_name]()
            else:
                return EMNIST_LETTERS_MODELS_MAPPING[model_name](**model_hparams)
    elif dataset_name == "digits":
        if model_name not in EMNIST_DIGITS_MODELS_MAPPING:
            raise ValueError(
                f"{model_name}: Invalid model name. Not supported for this dataset."
            )
        else:
            if not model_hparams:
                return EMNIST_DIGITS_MODELS_MAPPING[model_name]()
            else:
                return EMNIST_DIGITS_MODELS_MAPPING[model_name](**model_hparams)
    elif dataset_name == "mnist":
        if model_name not in EMNIST_MNIST_MODELS_MAPPING:
            raise ValueError(
                f"{model_name}: Invalid model name. Not supported for this dataset."
            )
        else:
            if not model_hparams:
                return EMNIST_MNIST_MODELS_MAPPING[model_name]()
            else:
                return EMNIST_MNIST_MODELS_MAPPING[model_name](**model_hparams)
    else:
        raise ValueError(
            f"{dataset_name}: Invalid dataset name. Not a supported dataset."
        )


###############
# End Utils #
###############


class BalancedEMNIST(pl.LightningModule):
    """PyTorch Lightning wrapper for EMNIST(balanced) dataset."""

    def __init__(
        self,
        model_name: EMNIST_MODELS_ENUM,
        optimizer_name: OPTIMIZERS_TYPE,
        optimizer_hparams: Dict[str, Any],
        model_hparams: Optional[Dict[str, Any]] = None,
        fl_hparams: Optional[FLParams] = None,
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
        self.model = create_model(
            dataset_name="balanced",
            model_name=model_name.value,
            model_hparams=model_hparams,
        )
        self.fl_hparams: Optional[Dict[str, Any]] = (
            fl_hparams.as_dict() if fl_hparams else None
        )
        combined_hparams: Dict[str, Any] = {
            "model_hparams": vars(self.model.hparams),
            "optimizer_hparams": {
                "optimizer_name": optimizer_name,
                "optimizer_fn": OPTIMIZERS_BY_NAME[optimizer_name.value],
                "config": optimizer_hparams,
            },
            "fl_hparams": vars(fl_hparams.as_simple_namespace()) if fl_hparams else {},
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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


class ByClassEMNIST(pl.LightningModule):
    """PyTorch Lightning wrapper for EMNIST(by class) dataset."""

    def __init__(
        self,
        model_name: EMNIST_MODELS_ENUM,
        optimizer_name: OPTIMIZERS_TYPE,
        optimizer_hparams: Dict[str, Any],
        model_hparams: Optional[Dict[str, Any]] = None,
        fl_hparams: Optional[FLParams] = None,
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
        self.model = create_model(
            dataset_name="byclass",
            model_name=model_name.value,
            model_hparams=model_hparams,
        )
        self.fl_hparams: Optional[Dict[str, Any]] = (
            fl_hparams.as_dict() if fl_hparams else None
        )
        combined_hparams: Dict[str, Any] = {
            "model_hparams": vars(self.model.hparams),
            "optimizer_hparams": {
                "optimizer_name": optimizer_name,
                "optimizer_fn": OPTIMIZERS_BY_NAME[optimizer_name.value],
                "config": optimizer_hparams,
            },
            "fl_hparams": vars(fl_hparams.as_simple_namespace()) if fl_hparams else {},
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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


class ByMergeEMNIST(pl.LightningModule):
    """PyTorch Lightning wrapper for EMNIST(by merge) dataset."""

    def __init__(
        self,
        model_name: EMNIST_MODELS_ENUM,
        optimizer_name: OPTIMIZERS_TYPE,
        optimizer_hparams: Dict[str, Any],
        model_hparams: Optional[Dict[str, Any]] = None,
        fl_hparams: Optional[FLParams] = None,
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
        self.model = create_model(
            dataset_name="bymerge",
            model_name=model_name.value,
            model_hparams=model_hparams,
        )
        self.fl_hparams: Optional[Dict[str, Any]] = (
            fl_hparams.as_dict() if fl_hparams else None
        )
        combined_hparams: Dict[str, Any] = {
            "model_hparams": vars(self.model.hparams),
            "optimizer_hparams": {
                "optimizer_name": optimizer_name,
                "optimizer_fn": OPTIMIZERS_BY_NAME[optimizer_name.value],
                "config": optimizer_hparams,
            },
            "fl_hparams": vars(fl_hparams.as_simple_namespace()) if fl_hparams else {},
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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


class LettersEMNIST(pl.LightningModule):
    """PyTorch Lightning wrapper for EMNIST(letters) dataset."""

    def __init__(
        self,
        model_name: EMNIST_MODELS_ENUM,
        optimizer_name: OPTIMIZERS_TYPE,
        optimizer_hparams: Dict[str, Any],
        model_hparams: Optional[Dict[str, Any]] = None,
        fl_hparams: Optional[FLParams] = None,
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
        self.model = create_model(
            dataset_name="letters",
            model_name=model_name.value,
            model_hparams=model_hparams,
        )
        self.fl_hparams: Optional[Dict[str, Any]] = (
            fl_hparams.as_dict() if fl_hparams else None
        )
        combined_hparams: Dict[str, Any] = {
            "model_hparams": vars(self.model.hparams),
            "optimizer_hparams": {
                "optimizer_name": optimizer_name,
                "optimizer_fn": OPTIMIZERS_BY_NAME[optimizer_name.value],
                "config": optimizer_hparams,
            },
            "fl_hparams": vars(fl_hparams.as_simple_namespace()) if fl_hparams else {},
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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


class DigitsEMNIST(pl.LightningModule):
    """PyTorch Lightning wrapper for EMNIST(digits) dataset."""

    def __init__(
        self,
        model_name: EMNIST_MODELS_ENUM,
        optimizer_name: OPTIMIZERS_TYPE,
        optimizer_hparams: Dict[str, Any],
        model_hparams: Optional[Dict[str, Any]] = None,
        fl_hparams: Optional[FLParams] = None,
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
        self.model = create_model(
            dataset_name="digits",
            model_name=model_name.value,
            model_hparams=model_hparams,
        )
        self.fl_hparams: Optional[Dict[str, Any]] = (
            fl_hparams.as_dict() if fl_hparams else None
        )
        combined_hparams: Dict[str, Any] = {
            "model_hparams": vars(self.model.hparams),
            "optimizer_hparams": {
                "optimizer_name": optimizer_name,
                "optimizer_fn": OPTIMIZERS_BY_NAME[optimizer_name.value],
                "config": optimizer_hparams,
            },
            "fl_hparams": vars(fl_hparams.as_simple_namespace()) if fl_hparams else {},
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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


class MNISTEMNIST(pl.LightningModule):
    """PyTorch Lightning wrapper for EMNIST(mnist) dataset."""

    def __init__(
        self,
        model_name: EMNIST_MODELS_ENUM,
        optimizer_name: OPTIMIZERS_TYPE,
        optimizer_hparams: Dict[str, Any],
        model_hparams: Optional[Dict[str, Any]] = None,
        fl_hparams: Optional[FLParams] = None,
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
        self.model = create_model(
            dataset_name="mnist",
            model_name=model_name.value,
            model_hparams=model_hparams,
        )
        self.fl_hparams: Optional[Dict[str, Any]] = (
            fl_hparams.as_dict() if fl_hparams else None
        )
        combined_hparams: Dict[str, Any] = {
            "model_hparams": vars(self.model.hparams),
            "optimizer_hparams": {
                "optimizer_name": optimizer_name,
                "optimizer_fn": OPTIMIZERS_BY_NAME[optimizer_name.value],
                "config": optimizer_hparams,
            },
            "fl_hparams": vars(fl_hparams.as_simple_namespace()) if fl_hparams else {},
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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
