import numpy as np

def avg_aggregate(params):
    """
    简单FedAvg聚合，均匀平均所有客户端参数。
    params: dict，key=client，value=模型参数(np.array 或类似结构)
    返回聚合后的模型参数
    """
    # 获取客户端数量
    num_clients = len(params)
    # 假设所有参数是同shape的numpy数组，先初始化一个全零数组
    aggregated = None
    for client_params in params.values():
        if aggregated is None:
            aggregated = np.zeros_like(client_params)
        aggregated += client_params
    aggregated /= num_clients
    return aggregated
