#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the pre-trained ResNet architectures using PyTorch and torchvision.

Contains:
    - ResNet18
    - ResNet34
    - ResNet50
    - ResNet101
    - ResNet152
    - ResNext50_32X4D
    - ResNext101_32X8D
    - WideResNet50_2
    - WideResNet101_2
"""

from torchvision import models
from types import SimpleNamespace
import torch.nn as nn
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNet18(models.ResNet):
    """ResNet18 base definition"""

    def __init__(
        self,
        pre_trained=True,
        feature_extract=True,
        num_classes=10,
        num_channels=3,
        act_fn_name="relu",
        **kwargs
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            - num_classes (int, optional): Number of classification outputs. Defaults to 10.
            - num_channels (int, optional): Number of incoming channels. Defaults to 3.
            - act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.hparams = SimpleNamespace(
            model_name="resnet18",
            pre_trained=pre_trained,
            feature_extract=bool(pre_trained and feature_extract),
            finetune=bool(not feature_extract),
            quantized=False,
            num_classes=num_classes,
            num_channels=num_channels,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if pre_trained:
            pretrained_model = models.resnet18(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ResNet34(models.ResNet):
    """ResNet34 base definition"""

    def __init__(
        self,
        pre_trained=True,
        feature_extract=True,
        num_classes=10,
        num_channels=3,
        act_fn_name="relu",
        **kwargs
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            - num_classes (int, optional): Number of classification outputs. Defaults to 10.
            - num_channels (int, optional): Number of incoming channels. Defaults to 3.
            - act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3])
        self.hparams = SimpleNamespace(
            model_name="resnet34",
            pre_trained=pre_trained,
            feature_extract=bool(pre_trained and feature_extract),
            finetune=bool(not feature_extract),
            quantized=False,
            num_classes=num_classes,
            num_channels=num_channels,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if pre_trained:
            pretrained_model = models.resnet34(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False
        if num_channels != 3:
            self.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ResNet50(models.ResNet):
    """ResNet50 base definition"""

    def __init__(
        self,
        pre_trained=True,
        feature_extract=True,
        num_classes=10,
        num_channels=3,
        act_fn_name="relu",
        **kwargs
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            - num_classes (int, optional): Number of classification outputs. Defaults to 10.
            - num_channels (int, optional): Number of incoming channels. Defaults to 3.
            - act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.hparams = SimpleNamespace(
            model_name="resnet50",
            pre_trained=pre_trained,
            feature_extract=bool(pre_trained and feature_extract),
            finetune=bool(not feature_extract),
            quantized=False,
            num_classes=num_classes,
            num_channels=num_channels,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if pre_trained:
            pretrained_model = models.resnet50(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False
        if num_channels != 3:
            self.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ResNet101(models.ResNet):
    """ResNet101 base definition"""

    def __init__(
        self,
        pre_trained=True,
        feature_extract=True,
        num_classes=10,
        num_channels=3,
        act_fn_name="relu",
        **kwargs
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            - num_classes (int, optional): Number of classification outputs. Defaults to 10.
            - num_channels (int, optional): Number of incoming channels. Defaults to 3.
            - act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(ResNet101, self).__init__(Bottleneck, [3, 4, 23, 3])
        self.hparams = SimpleNamespace(
            model_name="resnet101",
            pre_trained=pre_trained,
            feature_extract=bool(pre_trained and feature_extract),
            finetune=bool(not feature_extract),
            quantized=False,
            num_classes=num_classes,
            num_channels=num_channels,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if pre_trained:
            pretrained_model = models.resnet101(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False
        if num_channels != 3:
            self.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ResNet152(models.ResNet):
    """ResNet152 base definition"""

    def __init__(
        self,
        pre_trained=True,
        feature_extract=True,
        num_classes=10,
        num_channels=3,
        act_fn_name="relu",
        **kwargs
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            - num_classes (int, optional): Number of classification outputs. Defaults to 10.
            - num_channels (int, optional): Number of incoming channels. Defaults to 3.
            - act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(ResNet152, self).__init__(Bottleneck, [3, 8, 36, 3])
        self.hparams = SimpleNamespace(
            model_name="resnet152",
            pre_trained=pre_trained,
            feature_extract=bool(pre_trained and feature_extract),
            finetune=bool(not feature_extract),
            quantized=False,
            num_classes=num_classes,
            num_channels=num_channels,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if pre_trained:
            pretrained_model = models.resnet152(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False
        if num_channels != 3:
            self.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ResNext50_32X4D(models.ResNet):
    """ResNet50_32X4D base definition"""

    def __init__(
        self,
        pre_trained=True,
        feature_extract=True,
        num_classes=10,
        num_channels=3,
        act_fn_name="relu",
        **kwargs
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            - num_classes (int, optional): Number of classification outputs. Defaults to 10.
            - num_channels (int, optional): Number of incoming channels. Defaults to 3.
            - act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(ResNext50_32X4D, self).__init__(
            block=Bottleneck, layers=[3, 4, 6, 3], groups=32, width_per_group=4
        )
        self.hparams = SimpleNamespace(
            model_name="resnext50_32x4d",
            pre_trained=pre_trained,
            feature_extract=bool(pre_trained and feature_extract),
            finetune=bool(not feature_extract),
            quantized=False,
            num_classes=num_classes,
            num_channels=num_channels,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if pre_trained:
            pretrained_model = models.resnext50_32x4d(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False
        if num_channels != 3:
            self.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ResNext101_32X8D(models.ResNet):
    """ResNet101_32X8D base definition"""

    def __init__(
        self,
        pre_trained=True,
        feature_extract=True,
        num_classes=10,
        num_channels=3,
        act_fn_name="relu",
        **kwargs
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            - num_classes (int, optional): Number of classification outputs. Defaults to 10.
            - num_channels (int, optional): Number of incoming channels. Defaults to 3.
            - act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(ResNext101_32X8D, self).__init__(
            block=Bottleneck, layers=[3, 4, 23, 3], groups=32, width_per_group=8
        )
        self.hparams = SimpleNamespace(
            model_name="resnext101_32x8d",
            pre_trained=pre_trained,
            feature_extract=bool(pre_trained and feature_extract),
            finetune=bool(not feature_extract),
            quantized=False,
            num_classes=num_classes,
            num_channels=num_channels,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if pre_trained:
            pretrained_model = models.resnext101_32x8d(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False
        if num_channels != 3:
            self.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class WideResNet50_2(models.ResNet):
    """WideResNet50_2 base definition"""

    def __init__(
        self,
        pre_trained=True,
        feature_extract=True,
        num_classes=10,
        num_channels=3,
        act_fn_name="relu",
        **kwargs
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            - num_classes (int, optional): Number of classification outputs. Defaults to 10.
            - num_channels (int, optional): Number of incoming channels. Defaults to 3.
            - act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(WideResNet50_2, self).__init__(
            block=Bottleneck, layers=[3, 4, 6, 3], width_per_group=128
        )
        self.hparams = SimpleNamespace(
            model_name="wide_resnet50_2",
            pre_trained=pre_trained,
            feature_extract=bool(pre_trained and feature_extract),
            finetune=bool(not feature_extract),
            quantized=False,
            num_classes=num_classes,
            num_channels=num_channels,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if pre_trained:
            pretrained_model = models.wide_resnet50_2(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False
        if num_channels != 3:
            self.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class WideResNet101_2(models.ResNet):
    """WideResNet101_2 base definition"""

    def __init__(
        self,
        pre_trained=True,
        feature_extract=True,
        num_classes=10,
        num_channels=3,
        act_fn_name="relu",
        **kwargs
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            - num_classes (int, optional): Number of classification outputs. Defaults to 10.
            - num_channels (int, optional): Number of incoming channels. Defaults to 3.
            - act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(WideResNet101_2, self).__init__(
            block=Bottleneck, layers=[3, 4, 23, 3], width_per_group=128
        )
        self.hparams = SimpleNamespace(
            model_name="wide_resnet101_2",
            pre_trained=pre_trained,
            feature_extract=bool(pre_trained and feature_extract),
            finetune=bool(not feature_extract),
            quantized=False,
            num_classes=num_classes,
            num_channels=num_channels,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if pre_trained:
            pretrained_model = models.wide_resnet101_2(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False
        if num_channels != 3:
            self.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)
