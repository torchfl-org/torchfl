#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implementation of the Benchmark class to test the models in the non-federated setting."""

from torchfl.compatibility import OPTIMIZERS
from torch.nn import Module
from torchfl.logger import Logger
from typing import Optional, Dict, Union
from torch import cuda, device, Tensor, zeros, no_grad
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cross_entropy


class Benchmark:
    """Benchmark class to test the models in the non-federated setting."""

    def __init__(
        self,
        experiment_name: str,
        train_dataset: Dataset,
        test_dataset: Dataset,
        model: Module,
        optimizer: OPTIMIZERS,  # type: ignore
        logger: Optional[Logger] = None,
        epochs: int = 10,
        train_batch_size: int = 128,
        test_batch_size: int = 10,
        use_gpu: bool = False,
        save_model_path: Optional[str] = None,
        load_model_path: Optional[str] = None,
        console_out: bool = True,
        verbose: bool = False,
    ) -> None:
        """Constructor

        Args:
            experiment_name (str): name of the experiment for logging and storage purpose.
            train_dataset (Dataset): PyTorch Dataset object to be used for training.
            test_dataset (Dataset): PyTorch Dataset object to be used for testing.
            model (Module): PyTorch model to be used for training.
            optimizer (OPTIMIZERS): PyTorch optimizer to be used for training.
            logger (Optional[Logger], optional): logger object to record the metrics. Defaults to None.
            epochs (int, optional): number of epochs for training. Defaults to 10.
            train_batch_size (int, optional): batch size to use while training. Defaults to 128.
            test_batch_size (int, optional): batch size to use while testing. Defaults to 10.
            use_gpu (bool, optional): attempt to use the gpu, if available. Defaults to False.
            save_model_path (Optional[str], optional): save the trained model at the given path. Defaults to None.
            load_model_path (Optional[str], optional): load the model from the given path and start training. Defaults to None.
            console_out (bool, optional): allow output to console (stdout). Defaults to True.
            verbose (bool, optional): verbose output to console. Defaults to False.
        """
        self.experiment_name: str = experiment_name
        self.use_cuda: bool = bool(use_gpu and cuda.is_available())
        self.torch_device: device = device("cuda" if self.use_cuda else "cpu")
        self.kwargs: Dict[str, Union[str, int, bool]] = dict()
        if self.use_cuda:
            self.kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        self.train_loader: DataLoader = DataLoader(
            dataset=train_dataset, batch_size=train_batch_size, **self.kwargs
        )
        self.test_loader: DataLoader = DataLoader(
            dataset=test_dataset, batch_size=test_batch_size, **self.kwargs
        )
        self.model: Module = model
        self.optimizer: OPTIMIZERS = optimizer  # type: ignore
        self.logger: Optional[Logger] = logger
        self.epochs: int = epochs
        self.save_model_path: Optional[str] = save_model_path
        self.load_model_path: Optional[str] = load_model_path
        self.console_out: bool = console_out
        self.verbose: bool = bool(console_out and verbose)

    def train(self, epoch: int) -> None:
        """Training sub-routine"""
        self.model.train()
        train_loss: Tensor = zeros(1)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.torch_device), target.to(self.torch_device)
            self.optimizer.zero_grad()  # type: ignore
            output: Tensor = self.model(data)
            train_loss = cross_entropy(output, target)
            train_loss.backward()
            self.optimizer.step()  # type: ignore
            if self.verbose:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        train_loss.item(),
                    )
                )

    def test(self) -> None:
        """Testing sub-routine"""
        self.model.eval()
        test_loss: float = float(0)
        correct: int = 0

        with no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.torch_device), target.to(self.torch_device)
                output: Tensor = self.model(data)
                test_loss += float(
                    cross_entropy(output, target, reduction="sum").item()
                )
                pred: Tensor = output.argmax(dim=1, keepdim=True)
                correct += int(pred.eq(target.view_as(pred)).sum().item())
            test_loss /= len(self.test_loader.dataset)
            self.custom_print(
                "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss,
                    correct,
                    len(self.test_loader.dataset),
                    100.0 * correct / len(self.test_loader.dataset),
                )
            )

    def run(self, epoch: int) -> None:
        """Wrapper for benchmarking"""
        self.custom_print("Training begins...")
        for epoch in range(1, self.epochs + 1):
            self.custom_print(f"Running Epoch:\t{epoch}/{self.epochs}.")
            self.train(epoch)
            self.test()

    def custom_print(self, content: str) -> None:
        """Helper method for printing to console.

        Args:
            content (str): content to be printed.
        """
        if self.console_out:
            print(content)
