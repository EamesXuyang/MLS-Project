from .avg_aggregate import avg_aggregate
from  .median_aggregate import median_aggregate
from .prox_aggregate import prox_aggregate
from .trimmed_mean_aggregate import trimmed_mean_aggregate
from .weighed_avg_aggregate import weighted_avg_aggregate


aggregate_funcs = {
    'avg': avg_aggregate,
    'median': median_aggregate,
    'prox': prox_aggregate,
    'trimmed_mean': trimmed_mean_aggregate,
    'weighted_avg': weighted_avg_aggregate
}
