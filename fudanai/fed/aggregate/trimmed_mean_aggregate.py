import numpy as np
from ...tensor import Tensor

def trimmed_mean_aggregate(params, trim_ratio=0.2):
    aggregated = {}
    num_clients = len(params)
    trim_k = int(num_clients * trim_ratio)

    for key in next(iter(params.values())).keys():
        param_stack = np.stack([params[c][key].data for c in params], axis=0)  # [n, ...]
        sorted_params = np.sort(param_stack, axis=0)
        trimmed = sorted_params[trim_k:num_clients - trim_k]
        trimmed_mean = np.mean(trimmed, axis=0)
        aggregated[key] = Tensor(trimmed_mean)

    return aggregated
