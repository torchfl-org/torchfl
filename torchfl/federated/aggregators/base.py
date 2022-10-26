#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Aggregator class used in FL."""


class BaseAggregator:
    """BaseAggregator class used in FL."""

    def __init__(self) -> None:
        """Constructor."""
        pass

    def aggregate(self) -> None:
        """Aggregate the weights of the agents."""
        raise NotImplementedError
