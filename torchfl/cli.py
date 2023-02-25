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
    model: _ArgumentGroup = parser.add_argument_group("model")
    optimizer: _ArgumentGroup = parser.add_argument_group("optimizer")
    parser.add_argument_group("trainer")
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
    general.add_argument("-v", "--verbose", action="store_true")

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
    model.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="name of the model.",
    )
    # FIXME: Write sanitizer validation functions for all of the following arguments
    model.add_argument("--num_classes", type=int, help="number of classes.")
    model.add_argument("--num_channels", type=int, help="number of channels.")
    model.add_argument(
        "--activation_fn", type=str, help="activation function name."
    )
    model.add_argument("--pretrained", action="store_false")
    model.add_argument("--feature_extract", action="store_false")

    # optimizer args
    # FIXME: Write sanitizer validation functions for all of the following arguments
    optimizer.add_argument(
        "--optimizer_name",
        type=str,
        required=True,
        help="name of the optimizer.",
    )
    optimizer.add_argument("--lr", type=float, help="learning rate.")
    # NOTE: Optimizers support many more arguments. We can't support all of them in CLI, but we will add support for them in YAML configs.
    # NOTE: PyTorch Lightning has fancy features for optimizers configuration, but we don't support them via CLI or YAML configs.
    # We can explore adding them in the future.

    # trainer args
    ## FIXME - add arguments for the trainer

    # federated args
    federated.add_argument(
        "--num_agents",
        type=int,
        default=10,
        help="number of agents for federated learning.",
    )
    federated.add_argument("--global_epochs", type=int, default=10)
    federated.add_argument("--local_epochs", type=int, default=5)
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
    federated.add_argument("--local_test_split", type=float, default=0.1)
    federated.add_argument("--local_train_bs", type=int, default=10)
    federated.add_argument("--local_test_bs", type=int, default=10)
    federated.add_argument(
        "--sampling_method", type=str, default="random"
    )  # FIXME: add choices here
    federated.add_argument("--sampling_ratio", type=float, default=0.1)
    federated.add_argument(
        "--agent_type", type=str, default="v1"
    )  # FIXME: add choices here
    federated.add_argument(
        "--aggregation_type", type=str, default="fedavg"
    )  # FIXME: add choices here

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
