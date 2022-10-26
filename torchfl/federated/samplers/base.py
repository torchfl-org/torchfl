#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Sampler class used in FL."""
from abc import ABCMeta, abstractmethod
from torchfl.federated.types import AgentsType
from typing import List


class BaseSampler(metaclass=ABCMeta):
    """BaseSampler class used in FL."""

    def __init__(self, all_agents: List[AgentsType]) -> None:
        """Constructor."""
        super().__init__()
        self.agents: List[AgentsType] = all_agents

    @abstractmethod
    def sample(self, num: int) -> List[AgentsType]:
        """
        Sample agents.

        Args:
            num: number of agents to sample

        Returns:
            List of sampled agents
        """
        raise NotImplementedError
