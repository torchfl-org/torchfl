#!/usr/bin/env python

"""V1 Agent class used in FL."""

from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from torchfl.federated.agents.base import BaseAgent
from torchfl.federated.fl_params import FLParams

pl.seed_everything(42)


class V1Agent(BaseAgent):
    """V1Agent class used in FL."""

    def __init__(
        self, id: int, data_shard: DataLoader, model: Any | None = None
    ) -> None:
        """Constructor."""
        super().__init__(id, data_shard, model)

    def train(
        self,
        trainer: pl.Trainer,
        fl_params: FLParams,
    ) -> Any:
        """
        Train the agent.

        Args:
            trainer (pl.Trainer): Trainer object used to train the model.
            fl_params (FLParams): FLParams object containing the FL parameters.
        """
        if self.model is None:
            raise ValueError(
                f"Model is not assigned to the agent with id={self.id}."
            )

        train_data_shard_len = int(
            len(self.data_shard.dataset)  # type:ignore
            * fl_params.local_train_split
        )
        test_data_shard_len = (
            len(self.data_shard.dataset) - train_data_shard_len  # type:ignore
        )
        train_data_shard, val_data_shard = random_split(
            self.data_shard.dataset,
            [train_data_shard_len, test_data_shard_len],
        )
        train_dataloader = DataLoader(
            train_data_shard,
            batch_size=fl_params.local_train_batch_size,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            val_data_shard,
            batch_size=fl_params.local_test_batch_size,
            shuffle=False,
        )

        trainer.fit(self.model, train_dataloader, val_dataloader)
        # test best model based on the validation and test set
        val_result: list[dict[str, float]] = trainer.test(
            self.model, dataloaders=val_dataloader, verbose=True
        )
        test_result: list[dict[str, float]] = trainer.test(
            self.model, dataloaders=val_dataloader, verbose=True
        )
        result: dict[str, float] = {  # type:ignore
            "test_acc": test_result[0][
                f"{fl_params.experiment_name}_test_acc"
            ],
            "val_acc": val_result[0][f"{fl_params.experiment_name}_test_acc"],
        }
        return self.model, result
