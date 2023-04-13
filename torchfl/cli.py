#!/usr/bin/env python

"""Console script for torchfl."""
import copy
import sys
from argparse import (
    SUPPRESS,
    Action,
    ArgumentError,
    ArgumentParser,
    Namespace,
    _ArgumentGroup,
)
from sys import exit

import yaml

from torchfl import __version__
from torchfl.compatibility import TORCHFL_DIR
from torchfl.config_resolver import ConfigResolver
from torchfl.datamodules.types import DATASET_GROUPS_MAP
from torchfl.federated.types import (
    AGENTS_TYPE,
    AGGREGATORS_TYPE,
    SAMPLERS_TYPE,
)


class DatasetNameSanitizer(Action):
    # FIXME: move this sanitizer to ConfigResolver
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
    parser.add_argument_group("pl_trainer")
    federated: _ArgumentGroup = parser.add_argument_group("federated learning")

    # general args
    general.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    general.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to the config file to create a torchfl experiment. Options from the configuration file will override the CLI arguments.",
    )  # FIXME - overrides other args when this is provided.
    general.add_argument(
        "--experiment_name",
        type=str,
        required=("--config" not in sys.argv),
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
        required=("--config" not in sys.argv),
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
        required=("--config" not in sys.argv),
        help="name of the optimizer.",
    )
    optimizer.add_argument("--lr", type=float, help="learning rate.")
    # NOTE: Optimizers support many more arguments. We can't support all of them in CLI, but we will add support for them in YAML configs.
    # NOTE: PyTorch Lightning has fancy features for optimizers configuration, but we don't support them via CLI or YAML configs.
    # We can explore adding them in the future.

    # trainer args
    # FIXME - add arguments for the trainer

    # federated args
    federated.add_argument(
        "--federated", action="store_true", help="enable federated learning."
    )
    federated.add_argument(
        "--num_agents",
        type=int,
        default=10,
        help="number of agents for federated learning.",
    )
    federated.add_argument(
        "--global_epochs",
        type=int,
        default=10,
        help="number of global epochs.",
    )
    federated.add_argument(
        "--local_epochs",
        type=int,
        default=5,
        help="number of local epochs for individual agents.",
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
    federated.add_argument(
        "--local_test_split",
        type=float,
        default=0.1,
        help="test split ratio for individual agents.",
    )
    federated.add_argument(
        "--local_train_bs",
        type=int,
        default=10,
        help="local training batch size for individual agents.",
    )
    federated.add_argument(
        "--local_test_bs",
        type=int,
        default=10,
        help="local testing batch size for individual agents.",
    )
    federated.add_argument(
        "--sampling_method",
        type=str,
        default="random",
        choices=SAMPLERS_TYPE,
        help="sampling method for selecting the agents.",
    )
    federated.add_argument("--sampling_ratio", type=float, default=0.1)
    federated.add_argument(
        "--agent_type",
        type=str,
        default="v1",
        choices=AGENTS_TYPE,
        help="type of agent to use for federated learning.",
    )
    federated.add_argument(
        "--aggregation_type",
        type=str,
        default="fedavg",
        choices=AGGREGATORS_TYPE,
        help="type of aggregation to use for federated learning.",
    )

    return parser.parse_args()


def config_parser(filename: str) -> Namespace:
    """
    Defines an argparse Namespace by parsing elements of the YAML config

    Returns: Namespace:
                Namespace of arguments belonging to the config
    """

    name_match = ["name"]
    output = {}
    """
    There is an issue with the coalescing of configs where elements in the config may be named relative to each respective parent.
    So something called model_name may be called name. This recursive rename solves this issue by renaming elements but this may not be sustainable in the long term.
    """

    def recursive_flatmap_rename(root: dict, parent_name=""):
        if len(root) == 0:
            return
        for element in root.keys():
            if type(root[element]) is dict:
                recursive_flatmap_rename(root[element], parent_name=element)
            elif element in name_match:
                output[parent_name + "_" + element] = root[element]
            else:
                output[element] = root[element]

    with open(filename) as yaml_file:
        config = yaml.safe_load(yaml_file)
        recursive_flatmap_rename(copy.deepcopy(config))

        return Namespace(**output)


def main():
    """Console script for torchfl."""
    args = cli_parser()
    if args.config:
        config_args = config_parser(args.config)
        # Takes the union of the two Namespaces.
        for k, v in vars(config_args).items():
            if v != vars(args).get(k, SUPPRESS):
                setattr(args, k, v)

    _ = ConfigResolver(args)
    # FIXME - the config resolver should return a torchfl job which can be run here
    print("Arguments: " + str(args))
    return 0


if __name__ == "__main__":
    exit(main())  # pragma: no cover
