#!/usr/bin/env python

"""Random Sampler class used in FL."""

import random
from typing import Any

from torchfl.federated.samplers.base import BaseSampler

random.seed(42)


class RandomSampler(BaseSampler):
    """RandomSampler class used in FL."""

    def __init__(self, all_agents: list[Any]) -> None:
        """Constructor."""
        super().__init__(all_agents=all_agents)

    def sample(self, num: int) -> list[Any]:
        """
        Sample agents.

        Args:
            num: number of agents to sample

        Returns:
            List of sampled agents
        """
        return random.sample(self.agents, num)
