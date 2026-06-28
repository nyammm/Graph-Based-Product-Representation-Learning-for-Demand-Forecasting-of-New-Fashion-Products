import numpy as np
from sklearn.metrics import mean_absolute_error


def get_score(gt, forecasts):
    eps=1e-8
    
    mae = mean_absolute_error(gt, forecasts)
    wape = np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / (np.sum(np.abs(gt)) + eps)
    
    adj_smape = np.mean(np.mean(np.abs(gt - forecasts) / (np.abs(gt) + np.abs(forecasts) + eps), axis=-1))
    
    sum_gt = np.sum(gt, axis=-1)
    sum_fc = np.sum(forecasts, axis=-1)
    accum_smape = np.mean(np.abs(sum_gt - sum_fc) / (np.abs(sum_gt) + np.abs(sum_fc) + eps))

    return mae, wape, adj_smape, accum_smape