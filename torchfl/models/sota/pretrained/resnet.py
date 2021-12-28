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
    - ResNet50_32x4D
    - ResNet101_32x8D
    - WideResNet50_2
    - WideResNet101_2
"""

from torchvision import models
from types import SimpleNamespace
import torch.nn as nn