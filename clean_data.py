#the usual suspects
import pandas as pd 
import numpy as np
import json
import copy
import time
from datetime import datetime 
import re

#import classes and functions
from pair_trading.preproc import alph_settings as alph, val_steps as val, alph_api_wrapper as wrap, threading, filters, make_dfs as df, helper
print(dir())

settings = alph(
    apikey = "MD27QTTVAK3AJBUQ", 
    site = "https://www.alphavantage.co/query?", 
    date_range = [datetime(2016, 8, 9), datetime(2023, 3, 30)]
    )

_, cofi_writer_path = helper.reader_paths(filetype = "*.json", name = "cofi")
_, dailydata_writer_path = helper.reader_paths(filetype = "*.json", name = "daily")

with open(cofi_writer_path, 'r') as fp:
    cofi_dict = json.load(fp)

with open(dailydata_writer_path, 'r') as fp:
    daily_data = json.load(fp)

start = settings.date_range[0] 
end = settings.date_range[1]

data_dict = helper.get_params(daily_data)
data_key = list(data_dict.keys())
for i in data_key: 
    if len(data_dict[i]) == 0:
        data_dict.pop(i)
    else: 
        data_dict[i] = pd.DataFrame(data_dict[i][1:], columns = ["timestamp", "price", "volume"])
        data_dict[i]["timestamp"] = pd.to_datetime(data_dict[i]["timestamp"])
        data_dict[i] = data_dict[i].query("@start <= timestamp <= @end")
price_pivot = df.pivot_data(data_dict, key = "price").pct_change(axis = 1).dropna(axis = 1)
vol_pivot = df.pivot_data(data_dict, key = "volume").dropna()
price_raw = df.pivot_data(data_dict, key = "price").dropna(axis = 1).T

bad_lst, cofi_clean = filters.cofi_filter(cofi_dict)
cofi_pivot = df.pivot_cofi(cofi_clean)

reader_path = helper.writer_path()
timestamp = re.sub("-", "_", str(datetime.today().date())) + re.sub(":", "_", str(datetime.today().time()))[:-7]
price_pivot.to_csv(reader_path+"price_pca"+timestamp+".csv")
vol_pivot.to_csv(reader_path+"vol_pca"+timestamp+".csv")
cofi_pivot.to_csv(reader_path+"cofi_pca"+timestamp+".csv")
price_pivot.to_csv(reader_path+"raw_prices"+timestamp+".csv")