
import numpy as np
import torch

def compute_log_returns(prices, annualize=True, trading_days=252):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    if annualize:
        log_ret *= trading_days
    return log_ret

def create_lagged_data(data, windowsize, tickers_len):
    x, y = [], []
    for i in range(len(data) - windowsize):
        feature = data[i:i+windowsize, :]
        target = data[i + windowsize, :tickers_len]
        x.append(feature)
        y.append(target)
    return torch.Tensor(np.array(x)), torch.Tensor(np.array(y))

def create_lagged_data_averages(data, windowsize, tickers_len):
    x, y = [], []
    for i in range(len(data) - windowsize):
        feature = data[i:i+windowsize, :]
        target = np.mean(data[i+windowsize:min(i+windowsize+15,len(data)), :tickers_len], axis=0)
        x.append(feature)
        y.append(target)
    return torch.Tensor(np.array(x)), torch.Tensor(np.array(y))
