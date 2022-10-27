#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""An example script for using FL entrypoint to setup a FL experiment on MNIST."""

from torchfl.federated.entrypoint import Entrypoint
from torchfl.federated.fl_params import FLParams
from torchfl.federated.agents.v1 import V1Agent
from torchfl.federated.aggregators.fedavg import FedAvgAggregator
from torchfl.federated.samplers.random import RandomSampler

from torchfl.datamodules.emnist import EMNISTDataModule, SUPPORTED_DATASETS_TYPE
from torchfl.models.wrapper.emnist import MNISTEMNIST, EMNIST_MODELS_ENUM
from torchfl.compatibility import OPTIMIZERS_TYPE, TORCHFL_DIR

from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Dict, List


def initialize_agents(
    fl_params: FLParams, agent_data_shard_map: Dict[int, DataLoader]
) -> List[V1Agent]:
    """Initialize agents."""
    agents = []
    for agent_id in range(fl_params.num_agents):
        agent = V1Agent(
            id=agent_id,
            model=MNISTEMNIST(
                model_name=EMNIST_MODELS_ENUM.LENET,
                optimizer_name=OPTIMIZERS_TYPE.ADAM,
                optimizer_hparams={"lr": 0.001},
                fl_hparams=fl_params,
            ),
            data_shard=agent_data_shard_map[agent_id],
        )
        agents.append(agent)
    return agents


def get_agent_data_shard_map() -> EMNISTDataModule:
    # prepare the dataset
    datamodule: EMNISTDataModule = EMNISTDataModule(
        dataset_name=SUPPORTED_DATASETS_TYPE.MNIST,
        train_transforms=transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
            ]
        ),
        val_transforms=transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
            ]
        ),
        predict_transforms=transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
            ]
        ),
        test_transforms=transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
            ]
        ),
        train_batch_size=10,
    )
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def main() -> None:
    """Main function."""
    fl_params = FLParams(
        experiment_name="iid_mnist_fedavg_10_agents",
        num_agents=10,
        global_epochs=10,
        local_epochs=5,
        sampling_ratio=0.1,
    )
    global_model = MNISTEMNIST(
        model_name=EMNIST_MODELS_ENUM.LENET,
        optimizer_name=OPTIMIZERS_TYPE.ADAM,
        optimizer_hparams={"lr": 0.001},
        fl_hparams=fl_params,
    )
    agent_data_shard_map = get_agent_data_shard_map().federated_iid_dataloader(
        num_workers=fl_params.num_agents,
        workers_batch_size=fl_params.local_train_batch_size,
    )
    all_agents = initialize_agents(fl_params, agent_data_shard_map)
    entrypoint = Entrypoint(
        global_model=global_model,
        global_datamodule=get_agent_data_shard_map(),
        fl_hparams=fl_params,
        agents=all_agents,
        aggregator=FedAvgAggregator(all_agents=all_agents),
        sampler=RandomSampler(all_agents=all_agents),
    )
    entrypoint.run()


if __name__ == "__main__":
    main()
