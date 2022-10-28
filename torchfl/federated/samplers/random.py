#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Random Sampler class used in FL."""

import random
from torchfl.federated.samplers.base import BaseSampler
from typing import Any, List

random.seed(42)


class RandomSampler(BaseSampler):
    """RandomSampler class used in FL."""

    def __init__(self, all_agents: List[Any]) -> None:
        """Constructor."""
        super().__init__(all_agents=all_agents)

    def sample(self, num: int) -> List[Any]:
        """
        Sample agents.

        Args:
            num: number of agents to sample

        Returns:
            List of sampled agents
        """
        return random.sample(self.agents, num)
