#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Agent class used in FL."""

from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader
from torchfl.federated.fl_params import FLParams
from typing import Optional, Any

import pytorch_lightning as pl


class BaseAgent(metaclass=ABCMeta):
    """BaseAgent class used in FL."""

    def __init__(
        self, id: int, data_shard: DataLoader, model: Optional[Any] = None
    ) -> None:
        """Constructor."""
        self.id: int = id
        self.data_shard: DataLoader = data_shard
        self.model: Optional[Any] = model

    def assign_model(self, model: Any) -> None:
        """Assign a model to the agent."""
        self.model.load_state_dict(model.state_dict())

    def assign_data_shard(self, data_shard: DataLoader) -> None:
        """Assign a data shard to the agent."""
        self.data_shard = data_shard

    @abstractmethod
    def train(
        self,
        trainer: pl.Trainer,
        fl_params: FLParams,
    ) -> None:
        """Train the agent."""
        raise NotImplementedError
