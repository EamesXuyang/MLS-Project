import numpy as np
from ...tensor import Tensor

def weighted_avg_aggregate(params, weights):
    """weights 是 dict：client_id -> 样本数"""
    total_weight = sum(weights.values())
    aggregated = {}

    for key in next(iter(params.values())).keys():
        aggregated[key] = None
        for client in params:
            client_param = params[client][key].data
            weight = weights[client] / total_weight
            if aggregated[key] is None:
                aggregated[key] = np.zeros_like(client_param)
            aggregated[key] += weight * client_param
        aggregated[key] = Tensor(aggregated[key])

    return aggregated
