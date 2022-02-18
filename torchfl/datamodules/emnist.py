#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""PyTorch Lightning DataModule for EMNIST (balanced, byclass, bymerge, digits, letters, mnist) dataset.

Raises:
    ValueError: The given dataset name is not supported. Supported: balanced, byclass, bymerge, digits, letters, mnist.

Returns:
    DatasetSplit: Implementation of PyTorch key-value based Dataset.
    EMNISTDataModule: PyTorch LightningDataModule for EMNIST datasets. Supports iid and non-iid splits.
"""
