#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Standardized definition of the FL parameters to be passed around in the federated learning experiment."""

import os
from typing import Any, Dict, Optional
from types import SimpleNamespace
from torchfl.compatibility import TORCHFL_DIR


class FLParams:
    """Standardized definition of the FL parameters to be passed around in the federated learning experiment."""

    def __init__(
        self,
        experiment_name: str,
        checkpoint_load_path: Optional[str] = None,
        checkpoint_save_path: str = os.path.join(TORCHFL_DIR, "runs"),
        num_agents: int = 10,
        global_epochs: int = 10,
        local_epochs: int = 5,
        local_train_split: float = 0.9,
        local_test_split: float = 0.1,
        local_train_batch_size: int = 10,
        local_test_batch_size: int = 10,
        sampling_ratio: float = 0.1,
    ) -> None:
        """
        Constructor

        Args:
            - experiment_name (str): Name of the experiment.
            - checkpoint_load_path (Optional[str]): Path to the checkpoint to load.
            - checkpoint_save_path (str): Path to the checkpoint to save.
            - num_agents (int): Number of agents participating in the FL experiment.
            - global_epochs (int): Number of global epochs.
            - local_epochs (int): Number of local epochs.
            - local_train_split (float): Fraction of the local dataset to be used for training.
            - local_test_split (float): Fraction of the local dataset to be used for testing.
            - local_train_batch_size (int): Batch size for training.
            - local_test_batch_size (int): Batch size for testing.
            - sampling_ratio (float): Fraction of the agents to be sampled for each global epoch.
        """
        self.experiment_name: str = experiment_name
        self.checkpoint_load_path: Optional[str] = checkpoint_load_path
        self.checkpoint_save_path: str = checkpoint_save_path
        self.num_agents: int = num_agents
        self.global_epochs: int = global_epochs
        self.local_epochs: int = local_epochs
        self.local_train_split: float = local_train_split
        self.local_test_split: float = local_test_split
        self.local_train_batch_size: int = local_train_batch_size
        self.local_test_batch_size: int = local_test_batch_size
        self.sampling_ratio: float = sampling_ratio

    def __str__(self) -> str:
        """String representation of the FLParams object."""
        return (
            f"FLParams(experiment_name={self.experiment_name}, "
            f"checkpoint_load_path={self.checkpoint_load_path}, "
            f"checkpoint_save_path={self.checkpoint_save_path}, "
            f"global_epochs={self.global_epochs}, "
            f"local_epochs={self.local_epochs}, "
            f"local_train_split={self.local_train_split}, "
            f"local_test_split={self.local_test_split}, "
            f"local_train_batch_size={self.local_train_batch_size}, "
            f"local_test_batch_size={self.local_test_batch_size}, "
            f"sampling_ratio={self.sampling_ratio})"
        )

    def __repr__(self) -> str:
        """String representation of the FLParams object."""
        return self.__str__()

    def as_dict(self) -> Dict[str, Any]:
        """Convert the FLParams object to a dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "checkpoint_load_path": self.checkpoint_load_path,
            "checkpoint_save_path": self.checkpoint_save_path,
            "num_agents": self.num_agents,
            "global_epochs": self.global_epochs,
            "local_epochs": self.local_epochs,
            "local_train_split": self.local_train_split,
            "local_test_split": self.local_test_split,
            "local_train_batch_size": self.local_train_batch_size,
            "local_test_batch_size": self.local_test_batch_size,
            "sampling_ratio": self.sampling_ratio,
        }

    def as_simple_namespace(self) -> SimpleNamespace:
        """Convert the FLParams object to a SimpleNamespace."""
        return SimpleNamespace(**self.as_dict())
