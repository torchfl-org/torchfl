#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the PyTorch Lightning wrapper modules for CIFAR10 and CIFAR100 dataset."""

import pytorch_lightning as pl
import torch.nn as nn
from torchfl.models.wrapper.mappings import CIFAR10_MODELS_MAPPING, CIFAR100_MODELS_MAPPING
from torchfl.compatibility import OPTIMIZERS

pl.seed_everything(42)


class CIFAR10(pl.LightningModule):
    """PyTorch Lightning wrapper for CIFAR10 dataset."""

    def __init__(self, model_name: str, optimizer: OPTIMIZERS) -> None:
        super().__init__()
        # FIXME from here - figuring out the hyperparameters part
        # FIXME - finish all the wrappers
        # FIXME - add the tests for models as organized
        # FIXME - jump to reorganizing the data modules for torch lightning
        # FIXME - finish the tests for the data loading modules
        # FIXME - move on to the training wrapper (see trello)


class CIFAR100(pl.LightningModule):
    """PyTorch Lightning wrapper for CIFAR10 dataset."""

    def __init__(self, model: str, optimizer: OPTIMIZERS) -> None:
        super().__init__()
        # FIXME from here - figuring out the hyperparameters part
