import os
from typing import Any, Dict, Iterable, List, Set, Tuple
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .compatibility import DATASETS_LITERAL

np.random.seed(42)


class DatasetSplit(Dataset):
    def __init__(self, dataset: Dataset, idxs: Iterable[int]) -> None:
        self.dataset: Dataset = dataset
        self.idxs: Iterable[int] = list(idxs)

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, label = self.dataset[self.idxs[index]]
        return image, label


class FLDataLoader:
    """Data Loader for Federated Learning"""

    def __init__(
        self,
        num_workers: int = 10,
        worker_bs: int = 10,
        worker_ep: int = 5,
        iid: bool = True,
        niid_factor: int = 2,
        dataset: DATASETS_LITERAL = "mnist",
        test_bs: int = 128,
    ) -> None:
        """Constructor
        Args:
            num_workers (int, optional): [description]. Defaults to 10.
            worker_bs (int, optional): [description]. Defaults to 10.
            worker_ep (int, optional): [description]. Defaults to 5.
            iid (bool, optional): [description]. Defaults to True.
            niid_factor (int, optional): [description]. Defaults to 2.
            dataset (DATASETS_LITERAL, optional): [description]. Defaults to "mnist".
            test_bs (int, optional): [description]. Defaults to 128.
        """
        self.num_workers: int = num_workers
        self.worker_bs: int = worker_bs
        self.worker_ep: int = worker_ep
        self.iid: bool = iid
        self.niid_factor: int = niid_factor
        self.dataset: DATASETS_LITERAL = dataset
        self.test_bs: int = test_bs

    @staticmethod
    def load_dataset(name: DATASETS_LITERAL, training: bool) -> Dataset:
        ROOT = os.path.join(os.pardir, "data")
        if name.lower() == "mnist":
            return datasets.MNIST(
                ROOT,
                train=training,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            )
        elif name.lower() == "emnist":
            return datasets.EMNIST(
                ROOT,
                split="digits",
                train=training,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            )

        elif name.lower() == "cifar10":
            return datasets.CIFAR10(
                ROOT,
                train=training,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                ),
            )

        elif name.lower() == "cifar100":
            return datasets.CIFAR100(
                ROOT,
                train=training,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                ),
            )

        else:
            raise ValueError(
                f"name: the dataset is not currently supported. Found: {name}"
            )

    def train_iid(self) -> Dict[int, Dataset]:
        dataset: Dataset = self.load_dataset(self.dataset, True)
        items: int = len(dataset) // self.num_workers
        distribution: np.ndarray = np.random.randint(
            low=0, high=len(dataset), size=(self.num_workers, items)
        )
        federated: Dict[int, Dataset] = dict()
        for i in range(len(distribution)):
            federated[i] = DataLoader(
                DatasetSplit(dataset, distribution[i]),
                batch_size=self.worker_bs,
                shuffle=True,
            )
        return federated

    def train_non_iid(self) -> Dict[int, Dataset]:
        dataset: Dataset = self.load_dataset(self.dataset, True)
        shards: int = self.num_workers * self.niid_factor
        items: int = len(dataset) // shards
        idx_shard: List[int] = list(range(shards))
        classes: np.ndarray = dataset.targets.numpy()
        idxs: Any = np.arange(len(dataset))

        idxs_labels: np.ndarray = np.vstack((idxs, classes))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs: np.ndarray = idxs_labels[0, :]
        distribution: Dict[int, np.ndarray] = {
            i: np.array([], dtype="int64") for i in range(self.num_workers)
        }

        while idx_shard:
            for i in range(self.num_workers):
                rand_set: Set[int] = set(
                    np.random.choice(idx_shard, self.niid_factor, replace=False)
                )
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    distribution[i] = np.concatenate(
                        (distribution[i], idxs[rand * items : (rand + 1) * items]),
                        axis=0,
                    )
        federated: Dict[int, Dataset] = dict()
        for i in distribution:
            federated[i] = DataLoader(
                DatasetSplit(dataset, distribution[i]),
                batch_size=self.worker_bs,
                shuffle=True,
            )
        return federated

    def test(self) -> Dataset:
        dataset: Dataset = self.load_dataset(self.dataset, False)
        return DataLoader(dataset, batch_size=self.test_bs, shuffle=True)


# driver for testing
if __name__ == "__main__":
    from cli import cli_parser

    args = cli_parser()
    loader = FLDataLoader()
    print(loader.train_non_iid())
