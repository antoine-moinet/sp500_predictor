
import numpy as np

def compute_sharpe(values, frequency=20):
    returns = (np.array(values[1:]) - np.array(values[:-1])) / np.array(values[:-1])
    mu_hat = np.mean(returns)
    sigma_hat = np.std(returns)
    return mu_hat / sigma_hat * np.sqrt(252 / frequency)
