
import numpy as np

def value_strategy_predictor(factors, prices, tickers, predictor, horizon, frequency, n_titles, initial_cap=1_000_000, shift=0):
    total_cap = initial_cap
    value = [total_cap]
    dates = factors.index.values
    rebalancing_dates = [dates[min(shift+(i+1)*frequency, len(dates)-1)] for i in range(len(dates) // frequency)]
    rebalancing_dates = list(dict.fromkeys(rebalancing_dates))
    units_old = np.zeros(len(tickers))
    set_cap = False

    for date in rebalancing_dates:
        ind = int(np.where(dates == date)[0][0])
        if ind < horizon:
            continue
        inputs = np.array(factors.loc[dates[ind-horizon+1:ind+1]])
        pred_date = predictor(inputs)
        prices_date = prices.loc[date]
        sort_ind = np.argsort(pred_date)
        long_ind = sort_ind[-n_titles:]
        short_ind = sort_ind[:n_titles]

        units = np.zeros(len(tickers))
        if set_cap:
            total_cap = value[-1] + np.sum(units_old * prices_date)

        units[long_ind] = total_cap / (2 * n_titles)
        units[short_ind] = -total_cap / (2 * n_titles)
        units = units / prices_date

        value.append(value[-1] + (np.sum(units_old * prices_date) - np.sum(units * prices_date)))
        units_old = units
        set_cap = True

    final_value = value[-1] + np.sum(units * prices.loc[dates[ind]])
    value.append(final_value)
    return value
