import numpy as np
from ...tensor import Tensor

def avg_aggregate(params):
    num_clients = len(params)
    aggregated = {}

    # 遍历每个参数名
    for key in next(iter(params.values())).keys():
        aggregated[key] = None
        for client in params:
            client_param = params[client][key].data
            if aggregated[key] is None:
                aggregated[key] = np.zeros_like(client_param)
            aggregated[key] += client_param
        aggregated[key] /= num_clients
        aggregated[key] = Tensor(aggregated[key])

    return aggregated
