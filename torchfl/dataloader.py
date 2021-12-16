from typing import Iterable, Set, Tuple, Any, Dict
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from argparse import Namespace

from compatibility import DATASETS_LITERAL

import numpy as np
import os

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


class CustomDataLoader:
    def __init__(self, args: Namespace, kwargs: Dict[Any, Any]) -> None:
        self.args: Namespace = args
        self.kwargs: Dict[Any, Any] = kwargs

    @staticmethod
    def load_dataset(name: DATASETS_LITERAL, training: bool) -> Dataset:
        ROOT = os.path.join(os.pardir, "data")
        if name.lower() == "mnist":
            return datasets.MNIST(ROOT, train=training, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          (0.1307,), (0.3081,))
                                  ]))
        elif name.lower() == "emnist":
            return datasets.EMNIST(ROOT, split='digits', train=training, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))

        elif name.lower() == "cifar10":
            return datasets.CIFAR10(ROOT, train=training, download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(
                                            32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010))
                                    ]))

        elif name.lower() == "cifar100":
            return datasets.CIFAR100(ROOT, train=training, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(
                                             32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                              (0.2023, 0.1994, 0.2010))
                                     ]))

        else:
            raise ValueError(
                f"name: the dataset is not currently supported. Found: {name}")

    def train_iid(self) -> Dict[int, Dataset]:
        dataset: Dataset = self.load_dataset(self.args.dataset, True)
        items: int = len(dataset) // self.args.num_workers
        distribution: np.ndarray = np.random.randint(
            low=0, high=len(dataset), size=(self.args.num_workers, items))
        federated: Dict[int, Dataset] = dict()
        for i in range(len(distribution)):
            federated[i] = DataLoader(DatasetSplit(
                dataset, distribution[i]), batch_size=self.args.worker_bs, shuffle=True, **self.kwargs)
        return federated

    def train_non_iid(self) -> Dict[int, Dataset]:
        dataset: Dataset = self.load_dataset(self.args.dataset, True)
        return

    def test(self) -> Dict[int, Dataset]:
        dataset: Dataset = self.load_dataset(self.args.dataset, False)
        return

# driver for testing
if __name__ == '__main__':
    from cli import cli_parser
    args = cli_parser()
    loader = CustomDataLoader(args=args, kwargs={})
    print(loader.train_iid())

