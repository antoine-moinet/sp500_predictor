import numpy as np
import torch

def compute_log_returns(prices, annualize=True, trading_days=252):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    if annualize:
        log_ret *= trading_days
    return log_ret

def create_lagged_and_split_data(data, windowsize, cutoff, model, carhart=True):
    """ Create rolling window data set of pairs of time series snippets for prediction: (returns,factors)  -> returns
        Each input (x) is a window of past data, and the output (y) is the next value you want to predict.
        Output Shape of x will be: (num_samples-windowsize, windowsize, num_features)
        Output Shape of y will be: (num_samples-windowsize, num_tickers)
    """
    x, y = [], []
    for i in range(len(data) - windowsize):
        feature = data[i:i+windowsize, :]
        target = data[i + windowsize, :]      
        if carhart:
            x.append(feature)
        else:
            x.append(feature[:,:-5])
        y.append(target[:-5])   
    if model == "rnn":
        return torch.Tensor(np.array(x[:cutoff])), torch.Tensor(np.array(y[:cutoff])), torch.Tensor(np.array(x[cutoff:])), torch.Tensor(np.array(y[cutoff:]))   
    elif model == "linear" or model == "dnn":
        return np.squeeze(np.array(x[:cutoff]),axis=1), np.array(y[:cutoff]), np.squeeze(np.array(x[cutoff:]),axis=1), np.array(y[cutoff:])

def create_lagged_and_split_data_averages(data, windowsize, cutoff):
    data_np = data.to_numpy()  # Much faster to index
    x, y = [], []
    for i in range(len(data_np) - windowsize - 15 + 1):
        feature = data_np[i:i+windowsize, :]
        target = np.mean(data_np[i+windowsize:i+windowsize+15, :-5], axis=0)
        x.append(feature)
        y.append(target)
    
    x = np.array(x)
    y = np.array(y)
    return (
        torch.tensor(x[:cutoff], dtype=torch.float32),
        torch.tensor(y[:cutoff], dtype=torch.float32),
        torch.tensor(x[cutoff:], dtype=torch.float32),
        torch.tensor(y[cutoff:], dtype=torch.float32),
    )


   