#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Sampler class used in FL."""
from abc import ABCMeta, abstractmethod
from typing import Any, List


class BaseSampler(metaclass=ABCMeta):
    """BaseSampler class used in FL."""

    def __init__(self, all_agents: List[Any]) -> None:
        """Constructor."""
        super().__init__()
        self.agents: List[Any] = all_agents

    @abstractmethod
    def sample(self, num: int) -> List[Any]:
        """
        Sample agents.

        Args:
            num: number of agents to sample

        Returns:
            List of sampled agents
        """
        raise NotImplementedError
