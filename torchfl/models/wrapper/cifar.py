#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the PyTorch Lightning wrapper modules for CIFAR10 and CIFAR100 dataset."""

import pytorch_lightning as pl
import torch.nn as nn
from torchfl.compatibility import OPTIMIZERS

pl.seed_everything(42)

class CIFAR10(pl.LightningModule):
    """PyTorch Lightning wrapper for CIFAR10 dataset."""

    def __init__(self, model: nn.Module, optimizer: OPTIMIZERS) -> None:
        super().__init__()
        # FIXME from here - figuring out the hyperparameters part
