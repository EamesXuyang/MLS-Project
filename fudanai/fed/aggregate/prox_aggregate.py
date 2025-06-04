import numpy as np
from ...tensor import Tensor

def prox_aggregate(params, global_params, mu=0.1):
    num_clients = len(params)
    aggregated = {}

    for key in next(iter(params.values())).keys():
        aggregated[key] = None
        for client in params:
            local_param = params[client][key].data
            global_param = global_params[key].data
            if aggregated[key] is None:
                aggregated[key] = np.zeros_like(local_param)
            prox_term = local_param - mu * (local_param - global_param)
            aggregated[key] += prox_term
        aggregated[key] /= num_clients
        aggregated[key] = Tensor(aggregated[key])

    return aggregated
