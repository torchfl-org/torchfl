#!/usr/bin/env python

"""Base Sampler class used in FL."""
from abc import ABCMeta, abstractmethod
from typing import Any


class BaseSampler(metaclass=ABCMeta):
    """BaseSampler class used in FL."""

    def __init__(self, all_agents: list[Any]) -> None:
        """Constructor."""
        super().__init__()
        self.agents: list[Any] = all_agents

    @abstractmethod
    def sample(self, num: int) -> list[Any]:
        """
        Sample agents.

        Args:
            num: number of agents to sample

        Returns:
            List of sampled agents
        """
        raise NotImplementedError
