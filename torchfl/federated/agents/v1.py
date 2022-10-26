#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""V1 Agent class used in FL."""

from torch.utils.data import DataLoader
from torchfl.federated.agents.base import BaseAgent
from typing import Any, Optional


class V1Agent(BaseAgent):
    """V1Agent class used in FL."""

    def __init__(
        self, id: int, data_shard: DataLoader, model: Optional[Any] = None
    ) -> None:
        """Constructor."""
        super().__init__(id, data_shard, model)

    def train(self) -> None:
        """Train the agent."""
        pass
