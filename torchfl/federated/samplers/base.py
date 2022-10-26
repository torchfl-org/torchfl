#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Sampler class used in FL."""


class BaseSampler:
    """BaseSampler class used in FL."""

    def __init__(self) -> None:
        """Constructor."""
        pass

    def sample(self) -> None:
        """Sample agents."""
        raise NotImplementedError
