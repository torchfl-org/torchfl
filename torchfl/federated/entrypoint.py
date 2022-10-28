#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Entry point for the federated learning experiment."""

from copy import deepcopy
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    DeviceStatsMonitor,
    ModelSummary,
    RichProgressBar,
    Timer,
)
import torch
from torchfl.compatibility import TORCHFL_DIR
from torchfl.federated.types import (
    AgentsType,
    AggregatorsType,
    SamplersType,
)
from torchfl.federated.fl_params import FLParams
from typing import Any, Dict, List


class Entrypoint:
    def __init__(
        self,
        global_model: Any,
        global_datamodule: Any,
        fl_hparams: FLParams,
        agents: List[AgentsType],
        aggregator: AggregatorsType,
        sampler: SamplersType,
    ) -> None:
        """
        Constructor.

        Args:
            - agents (List[AgentsType]): List of agents participating in the FL experiment.
            - aggregator (AggregatorsType): Aggregator used in the FL experiment.
            - sampler (SamplersType): Sampler used in the FL experiment.
            - global_model (Any): Global model used in the FL experiment.
            - global_datamodule (Any): Global datamodule used in the FL experiment.
            - fl_hparams (FLParams): Federated learning hyperparameters.
        """
        self.agents: List[AgentsType] = [agent for agent in agents]
        self.aggregator: AggregatorsType = aggregator
        self.sampler: SamplersType = sampler
        self.global_model: Any = global_model
        self.global_datamodule: Any = global_datamodule
        self.fl_params: FLParams = fl_hparams

    def gen_global_trainer(self) -> pl.Trainer:
        ROOT_DIR_PATH = os.path.join(
            TORCHFL_DIR, "fl_logs", f"{self.fl_params.experiment_name}"
        )
        return pl.Trainer(
            max_epochs=self.fl_params.local_epochs,
            accelerator="gpu" if torch.cuda.is_available() else None,
            auto_lr_find=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            num_nodes=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            num_processes=1,
            resume_from_checkpoint=self.fl_params.checkpoint_load_path,
            detect_anomaly=True,
            logger=[
                TensorBoardLogger(
                    name=f"{self.fl_params.experiment_name}_global",
                    save_dir=ROOT_DIR_PATH,
                ),
                CSVLogger(
                    save_dir=ROOT_DIR_PATH,
                    name=f"{self.fl_params.experiment_name}_global",
                ),
            ],
            callbacks=[
                LearningRateMonitor("epoch"),
                DeviceStatsMonitor(),
                ModelSummary(),
                RichProgressBar(leave=True),
                Timer(),
            ],
            enable_checkpointing=False,
        )

    def gen_agent_trainer(self, agent: AgentsType) -> pl.Trainer:
        ROOT_DIR_PATH = os.path.join(
            TORCHFL_DIR, "fl_logs", f"{self.fl_params.experiment_name}"
        )
        return pl.Trainer(
            max_epochs=self.fl_params.local_epochs,
            accelerator="gpu" if torch.cuda.is_available() else None,
            auto_lr_find=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            num_nodes=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            num_processes=1,
            resume_from_checkpoint=self.fl_params.checkpoint_load_path,
            detect_anomaly=True,
            logger=[
                TensorBoardLogger(
                    name=f"{self.fl_params.experiment_name}_agent_{agent.id}",
                    save_dir=ROOT_DIR_PATH,
                ),
                CSVLogger(
                    save_dir=ROOT_DIR_PATH,
                    name=f"{self.fl_params.experiment_name}_agent_{agent.id}",
                ),
            ],
            callbacks=[
                LearningRateMonitor("epoch"),
                DeviceStatsMonitor(),
                ModelSummary(),
                RichProgressBar(leave=True),
                Timer(),
            ],
            enable_checkpointing=False,
        )

    def run(self) -> None:
        """Run the federated learning experiment."""
        global_trainer = self.gen_global_trainer()
        # extract the fl params
        num_sampled: int = int(len(self.agents) * self.fl_params.sampling_ratio)

        for ep in range(1, self.fl_params.global_epochs + 1):
            print("Current global epoch: ", ep)
            # collecting agent weights
            agent_models_map: Dict[int, Any] = dict()
            sampled_agents: List[AgentsType] = self.sampler.sample(num=num_sampled)
            for i, agent in enumerate(sampled_agents):
                print(
                    "Current Epoch: ",
                    ep,
                    "Current Agent ID: ",
                    agent.id,
                    "Current Agent Index: ",
                    i,
                    "Total Agents: ",
                    len(sampled_agents),
                )
                agent_trained_model, agent_result = agent.train(
                    self.gen_agent_trainer(agent), self.fl_params
                )
                agent_models_map[agent.id] = agent_trained_model.state_dict()
                print(agent_result)
            self.global_model.load_state_dict(
                self.aggregator.aggregate(self.global_model, agent_models_map)
            )

            # global evaluation
            global_trainer.test(
                self.global_model,
                self.global_datamodule.test_dataloader(),
                verbose=True,
            )
            # share the new global model with all the agents
            for agent in self.agents:
                agent.assign_model(self.global_model)
