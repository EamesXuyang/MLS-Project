import numpy as np
from ...tensor import Tensor

def median_aggregate(params):
    aggregated = {}

    for key in next(iter(params.values())).keys():
        param_stack = np.stack([params[c][key].data for c in params], axis=0)  # shape: [num_clients, ...]
        median_param = np.median(param_stack, axis=0)
        aggregated[key] = Tensor(median_param)

    return aggregated
