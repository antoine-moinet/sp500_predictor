import numpy as np
import matplotlib.pyplot as plt

def value_strategy_predictor(X_test, dates, prices, frequency, predictor, n_titles, plot_=True):
    # horizon: past factors in a windowsize will be used for prediction

    total_cap = 1000000.        # Initial capital
    value = [total_cap]     # Keep track of the value of your portfolio
    n_dates = len(X_test)//frequency # compute number of dates for rebalancing
    units_old = np.zeros([X_test.shape[2]-5])   # Initialize "old units" to 0
    for i in range(n_dates):  
        date =  dates[i*frequency] # Access date of current iteration
        inputs = X_test[[i*frequency]] # Access past factors up to today
        pred_date = predictor(inputs) # Make predictions about future performance based on todays input
        prices_date = np.array(prices.loc[date]) # Access current prices
        sort_ind = np.argsort(pred_date.squeeze()) # Sort stocks according to predicted performance in ascending order.
        long_ind = sort_ind[X_test.shape[2]-5-n_titles:] # Indices of those stocks that performed best
        short_ind = sort_ind[:n_titles] # Indices of those stocks that performed worst
        units = np.zeros([X_test.shape[2]-5]) # Initialize units to 0
        total_cap = value[-1]+np.sum(units_old*prices_date) # Previous value + gains you make from selling stocks (or buying back) you bought (or sold) in previous period at current price
        units[long_ind] = total_cap/(2*n_titles) # Set equal weights for stocks that you buy
        units[short_ind] = -total_cap/(2*n_titles) # Set equal weights for stocks that you shortsell
        units = units/prices_date # Convert from proportion of wealth to actual units
        ## Update value: liquidate previous position, build current one.
        value.append(value[-1]+(np.sum(units_old*prices_date)-np.sum(units*prices_date)))
        ## Set variables for next iteration
        units_old = units
    ## At terminal time we liquidate the full position:
    value.append(value[-1]+(np.sum(units*prices.loc[dates[min(len(dates)-1,n_dates*frequency)]])))
    if plot_:
        plt.plot(value)
    return value, dates[ [i*frequency for i in range(n_dates)] + [min(len(dates)-1, n_dates*frequency)] ]  # Append last date to dates 