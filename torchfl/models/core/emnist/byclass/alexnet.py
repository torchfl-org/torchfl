#!/usr/bin/env python

"""Contains the AlexNet model implementations for EMNIST (by class) dataset."""

from torchfl.models.sota.alexnet import AlexNet as BaseAlexNet


class AlexNet(BaseAlexNet):
    def __init__(
        self, pre_trained=True, feature_extract=False, num_channels=1
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super().__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_channels=num_channels,
            num_classes=62,
            act_fn_name="relu",
        )
