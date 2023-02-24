#!/usr/bin/env python

"""Console script for torchfl."""
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from sys import exit

from torchfl import __version__
from torchfl.compatibility import DATASETS


def cli_parser() -> Namespace:
    """Defines the argument parser and returns the Namespace.

    Returns:
        Namespace: Namespace object from argparse
    """
    parser = ArgumentParser()
    general: _ArgumentGroup = parser.add_argument_group("general")
    parser.add_argument_group("datamodule")
    parser.add_argument_group("model")
    federated: _ArgumentGroup = parser.add_argument_group("federated learning")

    # general args
    general.add_argument(
        "--version", action="version", version=f"torchfl {__version__}"
    )
    general.add_argument(
        "--experiment_name",
        type=str,
        default="torchfl_experiment",
        help="name of the experiment.",
    )
    general.add_argument(
        "--experiment_version",
        type=float,
        default=0.1,
        help="version of the experiment.",
    )

    general.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help=f"name of the dataset to be used. Supported: {DATASETS}",
    )
    general.add_argument(
        "--test_bs",
        type=int,
        default=128,
        help="batch size used for the testing dataset.",
    )

    # datamodule args

    # model args

    # federated args
    federated.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="number of workers for federated learning.",
    )
    federated.add_argument(
        "--worker_bs",
        type=int,
        default=10,
        help="batch size of the dataset for workers training locally.",
    )
    federated.add_argument(
        "--worker_ep",
        type=int,
        default=5,
        help="number of epochs for the workers training locally.",
    )
    federated.add_argument(
        "--iid",
        action="store_true",
        help="whether the dataset follows iid distribution or not.",
    )
    federated.add_argument(
        "--niid_factor",
        type=int,
        default=2,
        help="max number of classes held by each niid agent. lower the number, more measure of non-iidness.",
    )

    args = parser.parse_args()
    return args


def main():
    """Console script for torchfl."""
    args = cli_parser()

    print("Arguments: " + str(args))
    print("Replace this message by putting your code into " "torchfl.cli.main")
    return 0


if __name__ == "__main__":
    exit(main())  # pragma: no cover
