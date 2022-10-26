#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Types used within the federated learning utilities."""

import enum
from torchfl.federated.agents.base import BaseAgent
from torchfl.federated.aggregators.base import BaseAggregator
from torchfl.federated.samplers.base import BaseSampler
from typing import Any, Union, Type

# enums
class AgentsEnum(enum.Enum):
    """Enum class for the supported agent types."""

    BASE = BaseAgent


class AggregatorsEnum(enum.Enum):
    """Enum class for the supported aggregator types."""

    BASE = BaseAggregator


class SamplersEnum(enum.Enum):
    """Enum class for the supported sampler types."""

    BASE = BaseSampler


# type aliases
AgentsType = Union[Type[BaseAgent], Type[Any]]
AggregatorsType = Union[Type[BaseAggregator], Type[Any]]
SamplersType = Union[Type[BaseSampler], Type[Any]]
