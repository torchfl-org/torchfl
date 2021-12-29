#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the model mappings for specific datasets."""

from torchfl.models.core.cifar.cifar10 import (
    alexnet as cifar10_alexnet,
    densenet as cifar10_densenet,
    lenet as cifar10_lenet,
    mobilenet as cifar10_mobilenet,
    resnet as cifar10_resnet,
    shufflenetv2 as cifar10_shufflenetv2,
    squeezenet as cifar10_squeezenet,
    vgg as cifar10_vgg,
)
from torchfl.models.core.cifar.cifar10.quantized import (
    mobilenet as quantized_cifar10_mobilenet,
    resnet as quantized_cifar10_resnet,
    shufflenetv2 as quantized_cifar10_shufflenetv2,
)

CIFAR10_MODELS_MAPPING = {
    "alexnet": cifar10_alexnet.AlexNet,
    "densenet121": cifar10_densenet.DenseNet121,
    "densenet161": cifar10_densenet.DenseNet161,
    "densenet169": cifar10_densenet.DenseNet169,
    "densenet201": cifar10_densenet.DenseNet201,
    "lenet": cifar10_lenet.LeNet,
    "mobilenetv2": cifar10_mobilenet.MobileNetV2,
    "mobilenetv3small": cifar10_mobilenet.MobileNetV3Small,
    "mobilenetv3large": cifar10_mobilenet.MobileNetV3Large,
    "resnet18": cifar10_resnet.ResNet18,
    "resnet34": cifar10_resnet.ResNet34,
    "resnet50": cifar10_resnet.ResNet50,
    "resnet101": cifar10_resnet.ResNet101,
    "resnet152": cifar10_resnet.ResNet152,
    "resnext50_32x4d": cifar10_resnet.ResNext50_32X4D,
    "resnext101_32x8d": cifar10_resnet.ResNext101_32X8D,
    "wideresnet50_2": cifar10_resnet.WideResNet50_2,
    "wideresnet101_2": cifar10_resnet.WideResNet101_2,
    "shufflenetv2_x0_5": cifar10_shufflenetv2.ShuffleNetv2_x0_5,
    "shufflenetv2_x1_0": cifar10_shufflenetv2.ShuffleNetv2_x1_0,
    "shufflenetv2_x1_5": cifar10_shufflenetv2.ShuffleNetv2_x1_5,
    "shufflenetv2_x2_0": cifar10_shufflenetv2.ShuffleNetv2_x2_0,
    "squeezenet1_0": cifar10_squeezenet.SqueezeNet1_0,
    "squeezenet1_1": cifar10_squeezenet.SqueezeNet1_1,
    "vgg11": cifar10_vgg.VGG11,
    "vgg11_bn": cifar10_vgg.VGG11_BN,
    "vgg13": cifar10_vgg.VGG13,
    "vgg13_bn": cifar10_vgg.VGG13_BN,
    "vgg16": cifar10_vgg.VGG16,
    "vgg16_bn": cifar10_vgg.VGG16_BN,
    "vgg19": cifar10_vgg.VGG19,
    "vgg19_bn": cifar10_vgg.VGG19_BN,
    "quantized.mobilenetv2": quantized_cifar10_mobilenet.MobileNetV2,
    "quantized.mobilenetv3large": quantized_cifar10_mobilenet.MobileNetV3Large,
    "quantized.resnet18": quantized_cifar10_resnet.ResNet18,
    "quantized.resnet50": quantized_cifar10_resnet.ResNet50,
    "quantized.resnext101_32x8d": quantized_cifar10_resnet.ResNext101_32X8D,
    "quantized.shufflenetv2_x0_5": quantized_cifar10_shufflenetv2.ShuffleNetv2_x0_5,
    "quantized.shufflenetv2_x1_0": quantized_cifar10_shufflenetv2.ShuffleNetv2_x1_0,
}

from torchfl.models.core.cifar.cifar100 import (
    alexnet as cifar100_alexnet,
    densenet as cifar100_densenet,
    lenet as cifar100_lenet,
    mobilenet as cifar100_mobilenet,
    resnet as cifar100_resnet,
    shufflenetv2 as cifar100_shufflenetv2,
    squeezenet as cifar100_squeezenet,
    vgg as cifar100_vgg,
)
from torchfl.models.core.cifar.cifar100.quantized import (
    mobilenet as quantized_cifar100_mobilenet,
    resnet as quantized_cifar100_resnet,
    shufflenetv2 as quantized_cifar100_shufflenetv2,
)

CIFAR100_MODELS_MAPPING = {
    "alexnet": cifar100_alexnet.AlexNet,
    "densenet121": cifar100_densenet.DenseNet121,
    "densenet161": cifar100_densenet.DenseNet161,
    "densenet169": cifar100_densenet.DenseNet169,
    "densenet201": cifar100_densenet.DenseNet201,
    "lenet": cifar100_lenet.LeNet,
    "mobilenetv2": cifar100_mobilenet.MobileNetV2,
    "mobilenetv3small": cifar100_mobilenet.MobileNetV3Small,
    "mobilenetv3large": cifar100_mobilenet.MobileNetV3Large,
    "resnet18": cifar100_resnet.ResNet18,
    "resnet34": cifar100_resnet.ResNet34,
    "resnet50": cifar100_resnet.ResNet50,
    "resnet101": cifar100_resnet.ResNet101,
    "resnet152": cifar100_resnet.ResNet152,
    "resnext50_32x4d": cifar100_resnet.ResNext50_32X4D,
    "resnext101_32x8d": cifar100_resnet.ResNext101_32X8D,
    "wideresnet50_2": cifar100_resnet.WideResNet50_2,
    "wideresnet101_2": cifar100_resnet.WideResNet101_2,
    "shufflenetv2_x0_5": cifar100_shufflenetv2.ShuffleNetv2_x0_5,
    "shufflenetv2_x1_0": cifar100_shufflenetv2.ShuffleNetv2_x1_0,
    "shufflenetv2_x1_5": cifar100_shufflenetv2.ShuffleNetv2_x1_5,
    "shufflenetv2_x2_0": cifar100_shufflenetv2.ShuffleNetv2_x2_0,
    "squeezenet1_0": cifar100_squeezenet.SqueezeNet1_0,
    "squeezenet1_1": cifar100_squeezenet.SqueezeNet1_1,
    "vgg11": cifar100_vgg.VGG11,
    "vgg11_bn": cifar100_vgg.VGG11_BN,
    "vgg13": cifar100_vgg.VGG13,
    "vgg13_bn": cifar100_vgg.VGG13_BN,
    "vgg16": cifar100_vgg.VGG16,
    "vgg16_bn": cifar100_vgg.VGG16_BN,
    "vgg19": cifar100_vgg.VGG19,
    "vgg19_bn": cifar100_vgg.VGG19_BN,
    "quantized.mobilenetv2": quantized_cifar100_mobilenet.MobileNetV2,
    "quantized.mobilenetv3large": quantized_cifar100_mobilenet.MobileNetV3Large,
    "quantized.resnet18": quantized_cifar100_resnet.ResNet18,
    "quantized.resnet50": quantized_cifar100_resnet.ResNet50,
    "quantized.resnext101_32x8d": quantized_cifar100_resnet.ResNext101_32X8D,
    "quantized.shufflenetv2_x0_5": quantized_cifar100_shufflenetv2.ShuffleNetv2_x0_5,
    "quantized.shufflenetv2_x1_0": quantized_cifar100_shufflenetv2.ShuffleNetv2_x1_0,
}