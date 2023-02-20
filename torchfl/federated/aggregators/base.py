#!/usr/bin/env python

"""Base Aggregator class used in FL."""
from abc import ABCMeta, abstractmethod
from typing import Any


class BaseAggregator(metaclass=ABCMeta):
    """BaseAggregator class used in FL."""

    def __init__(self, all_agents: list[Any]) -> None:
        """Constructor."""
        super().__init__()
        self.agents: list[Any] = all_agents

    @abstractmethod
    def aggregate(
        self, global_model: Any, agent_models_map: dict[int, Any]
    ) -> Any:
        """
        Aggregate the weights of the agents. Compute the new global model using agent_models_map and update the models of all the agents.

        Args:
            global_model: global model
            agent_models_map: map of agent id to agent model

        Returns:
            new global model
        """
        raise NotImplementedError
