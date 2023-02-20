#!/usr/bin/env python

"""An example script to test the FedAvg aggregation."""
from torchfl.compatibility import OPTIMIZERS_TYPE
from torchfl.federated.aggregators.fedavg import FedAvgAggregator
from torchfl.models.wrapper.emnist import EMNIST_MODELS_ENUM, MNISTEMNIST

if __name__ == "__main__":
    model = MNISTEMNIST(
        EMNIST_MODELS_ENUM.LENET,
        OPTIMIZERS_TYPE.ADAM,
        {"lr": 0.001},
        {},
    )
    a_map = {0: model, 1: model}
    agg = FedAvgAggregator([])
    out = agg.aggregate(model, a_map)
    model.load_state_dict(out)
