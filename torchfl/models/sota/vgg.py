#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the pre-trained VGG architectures using PyTorch and torchvision.

Contains:
    - VGG11
    - VGG11_BN
    - VGG13
    - VGG13_BN
    - VGG16
    - VGG16_BN
    - VGG19
    - VGG19_BN
"""

from torchvision import models
from types import SimpleNamespace
import torch.nn as nn
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME
from torchvision.models.vgg import make_layers, cfgs
from typing import List, Union, cast


def _custom_make_layers(
    cfg: List[Union[str, int]], batch_norm: bool = False, starting_channels: int = 3
) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = starting_channels
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG11(models.VGG):
    """VGG11 base definition"""

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
        super(VGG11, self).__init__(make_layers(cfg=cfgs["A"], batch_norm=False))
        self.hparams = SimpleNamespace(
            model_name="vgg11",
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
            pretrained_model = models.vgg11(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.features = _custom_make_layers(
                cfg=cfgs["A"], batch_norm=False, starting_channels=num_channels
            )

        in_features = self.classifier[6].in_features
        self.classifier[6] = nn.Linear(in_features, self.hparams.num_classes)


class VGG11_BN(models.VGG):
    """VGG11_BN base definition"""

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
        super(VGG11_BN, self).__init__(make_layers(cfg=cfgs["A"], batch_norm=True))
        self.hparams = SimpleNamespace(
            model_name="vgg11_bn",
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
            pretrained_model = models.vgg11_bn(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.features = _custom_make_layers(
                cfg=cfgs["A"], batch_norm=True, starting_channels=num_channels
            )

        in_features = self.classifier[6].in_features
        self.classifier[6] = nn.Linear(in_features, self.hparams.num_classes)


class VGG13(models.VGG):
    """VGG13 base definition"""

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
        super(VGG13, self).__init__(make_layers(cfg=cfgs["B"]))
        self.hparams = SimpleNamespace(
            model_name="vgg13",
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
            pretrained_model = models.vgg13(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.features = _custom_make_layers(
                cfg=cfgs["B"], batch_norm=False, starting_channels=num_channels
            )

        in_features = self.classifier[6].in_features
        self.classifier[6] = nn.Linear(in_features, self.hparams.num_classes)


class VGG13_BN(models.VGG):
    """VGG13_BN base definition"""

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
        super(VGG13_BN, self).__init__(make_layers(cfg=cfgs["B"], batch_norm=True))
        self.hparams = SimpleNamespace(
            model_name="vgg13_bn",
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
            pretrained_model = models.vgg13_bn(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.features = _custom_make_layers(
                cfg=cfgs["B"], batch_norm=True, starting_channels=num_channels
            )

        in_features = self.classifier[6].in_features
        self.classifier[6] = nn.Linear(in_features, self.hparams.num_classes)


class VGG16(models.VGG):
    """VGG16 base definition"""

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
        super(VGG16, self).__init__(make_layers(cfg=cfgs["D"]))
        self.hparams = SimpleNamespace(
            model_name="vgg16",
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
            pretrained_model = models.vgg16(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.features = _custom_make_layers(
                cfg=cfgs["D"], batch_norm=False, starting_channels=num_channels
            )

        in_features = self.classifier[6].in_features
        self.classifier[6] = nn.Linear(in_features, self.hparams.num_classes)


class VGG16_BN(models.VGG):
    """VGG16_BN base definition"""

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
        super(VGG16_BN, self).__init__(make_layers(cfg=cfgs["D"], batch_norm=True))
        self.hparams = SimpleNamespace(
            model_name="vgg16_bn",
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
            pretrained_model = models.vgg16_bn(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.features = _custom_make_layers(
                cfg=cfgs["D"], batch_norm=True, starting_channels=num_channels
            )

        in_features = self.classifier[6].in_features
        self.classifier[6] = nn.Linear(in_features, self.hparams.num_classes)


class VGG19(models.VGG):
    """VGG19 base definition"""

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
        super(VGG19, self).__init__(make_layers(cfg=cfgs["E"]))
        self.hparams = SimpleNamespace(
            model_name="vgg19",
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
            pretrained_model = models.vgg19(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.features = _custom_make_layers(
                cfg=cfgs["E"], batch_norm=False, starting_channels=num_channels
            )

        in_features = self.classifier[6].in_features
        self.classifier[6] = nn.Linear(in_features, self.hparams.num_classes)


class VGG19_BN(models.VGG):
    """VGG19_BN base definition"""

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
        super(VGG19_BN, self).__init__(make_layers(cfg=cfgs["E"], batch_norm=True))
        self.hparams = SimpleNamespace(
            model_name="vgg19_bn",
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
            pretrained_model = models.vgg19_bn(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.features = _custom_make_layers(
                cfg=cfgs["E"], batch_norm=True, starting_channels=num_channels
            )

        in_features = self.classifier[6].in_features
        self.classifier[6] = nn.Linear(in_features, self.hparams.num_classes)
