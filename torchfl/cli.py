#!/usr/bin/env python

"""Console script for torchfl."""
from argparse import (
    Action,
    ArgumentError,
    ArgumentParser,
    Namespace,
    _ArgumentGroup,
)
from sys import exit

from torchfl import __version__
from torchfl.compatibility import TORCHFL_DIR
from torchfl.datamodules.types import DATASET_GROUPS_MAP


class DatasetNameSanitizer(Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values not in DATASET_GROUPS_MAP[namespace.dataset_group]:
            raise ArgumentError(
                namespace.dataset_name,
                f"Invalid dataset name: {values}. Supported: {str(DATASET_GROUPS_MAP[namespace.dataset_group])}",
            )
        setattr(namespace, self.dest, values)


def cli_parser() -> Namespace:
    """Defines the argument parser and returns the Namespace.

    Returns:
        Namespace: Namespace object from argparse
    """
    parser = ArgumentParser(prog="torchfl", description="torchfl CLI")
    general: _ArgumentGroup = parser.add_argument_group("general")
    datamodule: _ArgumentGroup = parser.add_argument_group("datamodule")
    parser.add_argument_group("model")
    federated: _ArgumentGroup = parser.add_argument_group("federated learning")

    # general args
    general.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    general.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="name of the experiment.",
    )
    general.add_argument(
        "--experiment_version",
        type=float,
        default=0.1,
        help="version of the experiment. Default: 0.1",
    )

    # datamodule args
    datamodule.add_argument(
        "--data_dir",
        type=str,
        help=f"path to store the downloaded data-related files. Default: {TORCHFL_DIR}",
        default=TORCHFL_DIR,
    )
    datamodule.add_argument(
        "--dataset_group",
        type=str,
        choices=DATASET_GROUPS_MAP.keys(),
        help="name of the dataset group.",
    )
    datamodule.add_argument(
        "--dataset_name",
        type=str,
        action=DatasetNameSanitizer,
        help="name of the dataset (should belong to a data group).",
    )
    datamodule.add_argument(
        "--validation_split",
        type=float,
        help="validation split ratio. Default: 0.1",
        default=0.1,
    )
    datamodule.add_argument(
        "--train_bs",
        type=int,
        help="batch size used for the training dataset. Default: 128",
        default=128,
    )
    datamodule.add_argument(
        "--validation_bs",
        type=int,
        help="batch size used for the validation dataset. Default: 1",
        default=1,
    )
    datamodule.add_argument(
        "--test_bs",
        type=int,
        help="batch size used for the testing dataset. Default: 1",
        default=1,
    )
    datamodule.add_argument(
        "--predict_bs",
        type=int,
        help="batch size used for the prediction dataset. Default: 1",
        default=1,
    )
    # NOTE: we can't support the following args as of now: train_transforms, val_transforms, test_transforms, predict_transforms

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
