
import os
import pandas as pd
import numpy as np
from pathlib import Path
import requests, zipfile, io

def download_and_extract_data(url, extract_to="data_full"):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to download data.")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(extract_to)

def load_price_data(data_dir):
    file_path = Path(data_dir) / "sp500stocks.csv"
    data = pd.read_csv(file_path, index_col=0)
    return data

def load_carhart_factors(data_dir, log_ret_index):
    ff3_path = Path(data_dir) / "F-F_Research_Data_Factors_daily.csv"
    mom_path = Path(data_dir) / "F-F_Momentum_Factor_daily.csv"

    ff_3 = np.log(pd.read_csv(ff3_path, header=3, index_col=0, skipfooter=1, engine="python") / 100 + 1) * 252
    ff_3.index = pd.to_datetime(ff_3.index, format="%Y%m%d")
    ff_3 = ff_3.reindex(log_ret_index)

    mom = np.log(pd.read_csv(mom_path, header=11, index_col=0, skipfooter=1, engine="python") / 100 + 1) * 252
    mom.index = pd.to_datetime(mom.index, format="%Y%m%d")
    mom = mom.reindex(log_ret_index)

    carhart = ff_3.merge(mom, how="left", left_index=True, right_index=True)
    return ff_3, mom, carhart
