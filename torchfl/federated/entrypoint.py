#!/usr/bin/env python

"""Entry point for the federated learning experiment."""

import os
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelSummary,
    RichProgressBar,
    Timer,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from torchfl.compatibility import TORCHFL_DIR
from torchfl.federated.fl_params import FLParams
from torchfl.federated.types import AgentsType, AggregatorsType, SamplersType


class Entrypoint:
    def __init__(
        self,
        global_model: Any,
        global_datamodule: Any,
        fl_hparams: FLParams,
        agents: list[AgentsType],
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
        self.agents: list[AgentsType] = list(agents)
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
            accelerator="gpu" if torch.cuda.is_available() else "auto",
            # auto_lr_find=True,    # NOTE: this feature was deprecated and they introduced a callback for this. We can get rid of this now but should come up w alternative.
            benchmark=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=torch.cuda.device_count()
            if torch.cuda.is_available()
            else 1,
            num_nodes=torch.cuda.device_count()
            if torch.cuda.is_available()
            else 1,
            default_root_dir=self.fl_params.checkpoint_load_path,
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
        )

    def gen_agent_trainer(self, agent: AgentsType) -> pl.Trainer:
        ROOT_DIR_PATH = os.path.join(
            TORCHFL_DIR, "fl_logs", f"{self.fl_params.experiment_name}"
        )
        return pl.Trainer(
            max_epochs=self.fl_params.local_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "auto",
            benchmark=True,
            # auto_lr_find=True,    # NOTE: this feature was deprecated and they introduced a callback for this. We can get rid of this now but should come up w alternative.
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=torch.cuda.device_count()
            if torch.cuda.is_available()
            else 1,
            num_nodes=torch.cuda.device_count()
            if torch.cuda.is_available()
            else 1,
            default_root_dir=self.fl_params.checkpoint_load_path,
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
        )

    def run(self) -> None:
        """Run the federated learning experiment."""
        global_trainer = self.gen_global_trainer()
        # extract the fl params
        num_sampled: int = int(
            len(self.agents) * self.fl_params.sampling_ratio
        )

        for ep in range(1, self.fl_params.global_epochs + 1):
            print("Current global epoch: ", ep)
            # collecting agent weights
            agent_models_map: dict[int, Any] = {}
            sampled_agents: list[AgentsType] = self.sampler.sample(
                num=num_sampled
            )
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
                agent_trained_model, agent_result = agent.train(  # type: ignore
                    self.gen_agent_trainer(agent), self.fl_params
                )
                agent_models_map[agent.id] = agent_trained_model.state_dict()  # type: ignore
                print(agent_result)  # type: ignore
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
