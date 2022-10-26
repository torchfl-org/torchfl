#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Entry point for the federated learning experiment."""


# agent_data_shard_map: Dict[int, DataLoaders]
# sampler object
# aggregator object
# global model object
# for every sampled agent, create a trainer object and store it in a map
# perform the training, extract the weights from the map after every round, send it to aggregator
# all of this compatible with PyTorch Lightning loggers and stuff

from torchfl.federated.types import (
    AgentsType,
    AggregatorsType,
    SamplersType,
    AgentsEnum,
    AggregatorsEnum,
    SamplersEnum,
)
from typing import Any, Dict, List, Optional


class Entrypoint:
    def __init__(
        self,
        global_model: Any,
        agents: List[AgentsEnum],
        aggregator: AggregatorsEnum = AggregatorsEnum.BASE,
        sampler: SamplersEnum = SamplersEnum.BASE,
        fl_hparams: Dict[str, Any] = dict(),
    ) -> None:
        """
        Constructor.

        Args:
            - agents (List[AgentsType]): List of agents participating in the FL experiment.
            - aggregator (AggregatorsType): Aggregator used in the FL experiment.
            - sampler (SamplersType): Sampler used in the FL experiment.
            - global_model (Any): Global model used in the FL experiment.
            - fl_hparams (Dict[str, Any]): Federated learning hyperparameters.
        """
        self.agents: List[AgentsEnum] = [agent.value() for agent in agents]
        self.aggregator: AggregatorsEnum = aggregator.value()
        self.sampler: SamplersEnum = sampler.value()
        self.global_model: Any = global_model
        self.fl_params: Dict[str, Any] = fl_hparams

    def run(self) -> None:
        """Run the federated learning experiment."""
        for ep in range(1, self.fl_params.get("num_epochs", 10) + 1):
            sampled_agents = self.sampler.sample()  # type: ignore
            for agent in sampled_agents:
                agent.train()
            aggregated_weights = self.aggregator.aggregate()  # type: ignore
            self.global_model.set_weights(aggregated_weights)
