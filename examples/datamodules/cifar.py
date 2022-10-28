#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""An example script to demonstrate the usage of CIFAR-10/CIFAR-100 datasets."""

import csv
from collections import Counter

from torchfl.datamodules.cifar import CIFARDataModule, SUPPORTED_DATASETS_TYPE
from torchfl.compatibility import TORCHFL_DIR

from torch.utils.data import DataLoader
from typing import Dict


def cifar10_iid_distribution(num_agents: int) -> Dict[int, DataLoader]:
    """Return an iid distribution for the CIFAR10 dataset given the number of agents"""
    datamodule: CIFARDataModule = CIFARDataModule(
        dataset_name=SUPPORTED_DATASETS_TYPE.CIFAR10
    )
    datamodule.prepare_data()
    datamodule.setup()
    agent_shard_map: Dict[int, DataLoader] = datamodule.federated_iid_dataloader(
        num_workers=num_agents
    )
    return agent_shard_map


def cifar10_noniid_distribution(
    num_agents: int, niid_factor: int
) -> Dict[int, DataLoader]:
    """Return a non-iid distribution for CIFAR10 dataset given the number of agents and niid_factor."""
    datamodule: CIFARDataModule = CIFARDataModule(
        dataset_name=SUPPORTED_DATASETS_TYPE.CIFAR10
    )
    datamodule.prepare_data()
    datamodule.setup()
    agent_shard_map: Dict[int, DataLoader] = datamodule.federated_non_iid_dataloader(
        num_workers=num_agents, niid_factor=niid_factor
    )
    return agent_shard_map


def cifar100_iid_distribution(num_agents: int) -> Dict[int, DataLoader]:
    """Return an iid distribution for the CIFAR100 dataset given the number of agents"""
    datamodule: CIFARDataModule = CIFARDataModule(
        dataset_name=SUPPORTED_DATASETS_TYPE.CIFAR100
    )
    datamodule.prepare_data()
    datamodule.setup()
    agent_shard_map: Dict[int, DataLoader] = datamodule.federated_iid_dataloader(
        num_workers=num_agents
    )
    return agent_shard_map


def cifar100_noniid_distribution(
    num_agents: int, niid_factor: int
) -> Dict[int, DataLoader]:
    """Return a non-iid distribution for CIFAR100 dataset given the number of agents and niid_factor."""
    datamodule: CIFARDataModule = CIFARDataModule(
        dataset_name=SUPPORTED_DATASETS_TYPE.CIFAR100
    )
    datamodule.prepare_data()
    datamodule.setup()
    agent_shard_map: Dict[int, DataLoader] = datamodule.federated_non_iid_dataloader(
        num_workers=num_agents, niid_factor=niid_factor
    )
    return agent_shard_map


def dump_data_distribution_to_csv(
    agent_shard_map: Dict[int, DataLoader], file_path: str
):
    """Dump the data distribution to a csv file"""
    # for agent_id, agent_dataloader in agent_shard_map.items():
    #     print("FOO")
    #     print(agent_id)
    #     print("BAR")
    #     print(agent_dataloader.dataset.targets)
    #     print(len(agent_dataloader.dataset.targets))
    #     print(Counter(agent_dataloader.dataset.targets))
    with open(file_path, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "agent_id",
                "label_0",
                "label_1",
                "label_2",
                "label_3",
                "label_4",
                "label_5",
                "label_6",
                "label_7",
                "label_8",
                "label_9",
            ]
        )
        for agent_id, data_loader in agent_shard_map.items():
            holdings = Counter(data_loader.dataset.targets)
            writer.writerow(
                [
                    agent_id,
                    holdings[0],
                    holdings[1],
                    holdings[2],
                    holdings[3],
                    holdings[4],
                    holdings[5],
                    holdings[6],
                    holdings[7],
                    holdings[8],
                    holdings[9],
                ]
            )


if __name__ == "__main__":
    agent_shard_map_5_agents_iid: Dict[int, DataLoader] = cifar10_iid_distribution(
        num_agents=5
    )
    agent_shard_map_10_agents_iid: Dict[int, DataLoader] = cifar10_iid_distribution(
        num_agents=10
    )
    agent_shard_map_100_agents_iid: Dict[int, DataLoader] = cifar10_iid_distribution(
        num_agents=100
    )
    agent_shard_map_5_agents_non_iid_1: Dict[
        int, DataLoader
    ] = cifar10_noniid_distribution(num_agents=5, niid_factor=1)
    agent_shard_map_10_agents_non_iid_1: Dict[
        int, DataLoader
    ] = cifar10_noniid_distribution(num_agents=10, niid_factor=1)
    agent_shard_map_100_agents_non_iid_1: Dict[
        int, DataLoader
    ] = cifar10_noniid_distribution(num_agents=100, niid_factor=1)
    agent_shard_map_5_agents_non_iid_3: Dict[
        int, DataLoader
    ] = cifar10_noniid_distribution(num_agents=5, niid_factor=3)
    agent_shard_map_5_agents_non_iid_5: Dict[
        int, DataLoader
    ] = cifar10_noniid_distribution(num_agents=5, niid_factor=5)
    dump_data_distribution_to_csv(
        agent_shard_map_5_agents_non_iid_5, "data_distribution_5_agents_non_iid_5.csv"
    )
    agent_shard_map_10_agents_non_iid_3: Dict[
        int, DataLoader
    ] = cifar10_noniid_distribution(num_agents=10, niid_factor=3)
    agent_shard_map_100_agents_non_iid_3: Dict[
        int, DataLoader
    ] = cifar10_noniid_distribution(num_agents=100, niid_factor=3)
    # dump_data_distribution_to_csv(
    #     agent_shard_map_100_agents_non_iid_3, "data_distribution_100_agents_niid_3.csv"
    # )

    # cifar100 experiments
    cifar100_agent_shard_map_100_agents_iid: Dict[
        int, DataLoader
    ] = cifar100_iid_distribution(num_agents=1000)
    cifar100_agent_shard_map_100_agents_non_iid_1: Dict[
        int, DataLoader
    ] = cifar100_noniid_distribution(num_agents=1000, niid_factor=1)
    cifar100_agent_shard_map_100_agents_non_iid_3: Dict[
        int, DataLoader
    ] = cifar100_noniid_distribution(num_agents=1000, niid_factor=3)

    # get the number of unique labels held by each agent
    ctr_iid = Counter(cifar100_agent_shard_map_100_agents_iid[5].dataset.targets)
    ctr_niid_1 = Counter(
        cifar100_agent_shard_map_100_agents_non_iid_1[5].dataset.targets
    )
    ctr_niid_3 = Counter(
        cifar100_agent_shard_map_100_agents_non_iid_3[5].dataset.targets
    )
    unique_labels_iid = 0
    unique_labels_niid_1 = 0
    unique_labels_niid_3 = 0

    for k in ctr_iid.keys():
        if (k in ctr_iid) and (ctr_iid[k] > 0):
            unique_labels_iid += 1

    for k in ctr_niid_1.keys():
        if (k in ctr_niid_1) and (ctr_niid_1[k] > 0):
            unique_labels_niid_1 += 1

    for k in ctr_niid_3.keys():
        if (k in ctr_niid_3) and (ctr_niid_3[k] > 0):
            unique_labels_niid_3 += 1
    print(unique_labels_iid)
    print(unique_labels_niid_1)
    print(unique_labels_niid_3)

    # print(agent_shard_map_5_agents_non_iid_1)
    # print(agent_shard_map_10_agents_non_iid_1)
    # print(agent_shard_map_100_agents_non_iid_1)
