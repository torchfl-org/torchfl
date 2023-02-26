#!/usr/bin/env python

"""Types used within the federated learning utilities."""

import enum

from torchfl.federated.agents.base import BaseAgent
from torchfl.federated.agents.v1 import V1Agent
from torchfl.federated.aggregators.base import BaseAggregator
from torchfl.federated.aggregators.fedavg import FedAvgAggregator
from torchfl.federated.samplers.base import BaseSampler
from torchfl.federated.samplers.random import RandomSampler


# enums
class AgentsEnum(enum.Enum):
    """Enum class for the supported agent types."""

    BASE = BaseAgent
    V1 = V1Agent


class AggregatorsEnum(enum.Enum):
    """Enum class for the supported aggregator types."""

    BASE = BaseAggregator
    FEDAVG = FedAvgAggregator


class SamplersEnum(enum.Enum):
    """Enum class for the supported sampler types."""

    BASE = BaseSampler
    RANDOM = RandomSampler


# type aliases
AgentsType = BaseAgent | V1Agent
AggregatorsType = BaseAggregator | FedAvgAggregator
SamplersType = BaseSampler | RandomSampler

AGENTS_TYPE = ["v1"]
AGGREGATORS_TYPE = ["fedavg"]
SAMPLERS_TYPE = ["random"]
