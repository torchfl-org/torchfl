#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Types used within the federated learning utilities."""

import enum
from typing import Any, Union

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
AgentsType = Union[BaseAgent, V1Agent]
AggregatorsType = Union[BaseAggregator, FedAvgAggregator]
SamplersType = Union[BaseSampler, RandomSampler]
