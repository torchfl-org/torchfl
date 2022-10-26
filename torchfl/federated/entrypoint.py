#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Entry point for the federated learning experiment."""

from torchfl.federated.types import (
    AgentsEnum,
    AgentsType,
    AggregatorsEnum,
    AggregatorsType,
    SamplersEnum,
    SamplersType,
)
from typing import Any, Dict, List


class Entrypoint:
    def __init__(
        self,
        global_model: Any,
        agents: List[AgentsEnum],
        aggregator: AggregatorsEnum = AggregatorsEnum.FEDAVG,
        sampler: SamplersEnum = SamplersEnum.RANDOM,
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
        self.agents: List[AgentsType] = [agent.value() for agent in agents]
        self.aggregator: AggregatorsType = aggregator.value()
        self.sampler: SamplersType = sampler.value(all_agents=self.agents)
        self.global_model: Any = global_model
        self.fl_params: Dict[str, Any] = fl_hparams

    def run(self) -> None:
        """Run the federated learning experiment."""
        # extract the fl params
        global_epochs: int = self.fl_params.get("global_epochs", 10)
        local_epochs: int = self.fl_params.get("local_epochs", 5)
        sampling_ratio: float = self.fl_params.get("sampling_ratio", 0.1)
        num_sampled: int = int(len(self.agents) * sampling_ratio)

        for ep in range(1, global_epochs + 1):
            # collecting agent weights
            agent_models_map: Dict[int, Any] = dict()
            sampled_agents: List[AgentsType] = self.sampler.sample(num=num_sampled)
            for agent in sampled_agents:
                agent_models_map[agent.id] = agent.train()
            self.global_model.set_weights(
                self.aggregator.aggregate(self.global_model, agent_models_map)
            )
            # share the new global model with all the agents
            for agent in self.agents:
                agent.assign_model(self.global_model)
