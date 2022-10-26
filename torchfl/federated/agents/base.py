#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Agent class used in FL."""

from torch.utils.data import DataLoader
from typing import Optional, Any


class BaseAgent:
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
        self.model = model

    def assign_data_shard(self, data_shard: DataLoader) -> None:
        """Assign a data shard to the agent."""
        self.data_shard = data_shard

    def train(self) -> None:
        """Train the agent."""
        raise NotImplementedError
