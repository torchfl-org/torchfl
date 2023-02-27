#!/usr/bin/env python

"""Contains the implementation of the ConfigResolver class."""

from argparse import Namespace


class ConfigResolver:
    def __init__(self, config: Namespace) -> None:
        """Constructor

        Args:
            - config (Namespace): Namespace object from argparse
        """
        self.config = config
