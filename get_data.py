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

#get tickers that are valid across the past six years
tker_dict = val.get_tkers(settings)
valid_lst = val.get_valid_tkers(tker_dict, pretty_print = False)

#filtering the tickers we want for analysis 
raw_df = pd.DataFrame(valid_lst, columns = ["ticker", "name", "exchange", "type", "date", "delist", "status"])
type_filt = "Stock" 
exchange_filt = "NYSE"
raw_df = raw_df.query("type == @type_filt")
raw_df = raw_df.query("exchange == @exchange_filt")

#list of tickers to work on for now
ticker_lst = raw_df["ticker"].tolist()

cofi_lst = ['MarketCapitalization', 'DividendPerShare', 'DividendYield', 'EPS', 'RevenuePerShareTTM', 'ProfitMargin', 'OperatingMarginTTM', 'ReturnOnAssetsTTM', 'ReturnOnEquityTTM', 'RevenueTTM', 'GrossProfitTTM', 'DilutedEPSTTM', 'QuarterlyEarningsGrowthYOY', 'QuarterlyRevenueGrowthYOY', 'AnalystTargetPrice', 'TrailingPE', 'PriceToSalesRatioTTM', 'PriceToBookRatio', 'EVToRevenue', 'EVToEBITDA', 'Beta']
hit_lst = copy.deepcopy(ticker_lst)
time_it = 3
ticker_info_data = {}
while len(hit_lst) > 0 and time_it > 2:
    ticker_info_url = wrap.get_general_url(settings, ticker_lst = hit_lst, output = "lst", datatype = "csv", function = "OVERVIEW")
    hit_lst, time_it, data = threading.thrd_json_data(ticker_info_url, key_lst = cofi_lst, wrapper_function = wrap.get_json_data)
    ticker_info_data.update(data)
    print(hit_lst)
    time.sleep(10)

reader_path = helper.writer_path()
timestamp = re.sub("-", "_", str(datetime.today().date())) + re.sub(":", "_", str(datetime.today().time()))[:-7]

with open(reader_path+"NYSE_cofi"+timestamp+".json", 'w') as fp:
    json.dump(ticker_info_data, fp)

cofi_filter, filtered_cofi_dict = filters.cofi_filter(ticker_info_data)
clean_df = raw_df.query("ticker != @cofi_filter")

temp_lst = clean_df["ticker"].tolist()

#get daily data 
hit_lst_2 = copy.deepcopy(temp_lst)
time_it_2 = 3
daily_data = {}
while len(hit_lst_2) > 0 and time_it_2 > 2:
    daily_urls = wrap.get_general_url(settings, datatype = "csv", ticker_lst = hit_lst_2, output = "lst", function = "TIME_SERIES_DAILY_ADJUSTED")
    hit_lst_2, time_it_2, data = threading.thrd_csv_data(daily_urls)
    daily_data.update(data)
    print(hit_lst_2)
    time.sleep(15)
with open(reader_path+"NYSE_dailydata"+timestamp+".json", 'w') as fp:
    json.dump(daily_data, fp)