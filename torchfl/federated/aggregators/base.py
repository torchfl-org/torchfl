#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Aggregator class used in FL."""
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List


class BaseAggregator(metaclass=ABCMeta):
    """BaseAggregator class used in FL."""

    def __init__(self, all_agents: List[Any]) -> None:
        """Constructor."""
        super().__init__()
        self.agents: List[Any] = all_agents

    @abstractmethod
    def aggregate(self, global_model: Any, agent_models_map: Dict[int, Any]) -> Any:
        """
        Aggregate the weights of the agents. Compute the new global model using agent_models_map and update the models of all the agents.

        Args:
            global_model: global model
            agent_models_map: map of agent id to agent model

        Returns:
            new global model
        """
        raise NotImplementedError
