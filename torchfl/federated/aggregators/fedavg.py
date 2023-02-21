#!/usr/bin/env python

"""FedAvg Aggregator class used in FL."""

from collections import OrderedDict
from typing import Any

import torch

from torchfl.federated.aggregators.base import BaseAggregator


class FedAvgAggregator(BaseAggregator):
    """FedAvgAggregator class used in FL."""

    def __init__(self, all_agents: list[Any]) -> None:
        """Constructor."""
        super().__init__(all_agents)

    def aggregate(
        self, global_model: Any, agent_models_map: dict[int, Any]
    ) -> Any:
        """
        Aggregate the weights of the agents. Compute the new global model using agent_models_map and update the models of all the agents.

        Args:
            global_model (Any): Global model used in the FL experiment.
            agent_models_map (Dict[int, Any]): map of agent id to agent model

        Returns:
            new global model
        """
        w_avg: dict[Any, Any] = OrderedDict()
        for _, models in agent_models_map.items():
            for key in global_model.state_dict().keys():
                if key not in w_avg.keys():
                    w_avg[key] = models[key].clone()
                else:
                    w_avg[key] += models[key].clone()
        for key in w_avg.keys():
            w_avg[key] = torch.divide(w_avg[key], len(agent_models_map))
        return w_avg
