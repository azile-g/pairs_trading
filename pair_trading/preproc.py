#the usual suspects
import itertools
import statistics
import csv
import pandas as pd
import time
import os
import glob

#api related
from urllib.parse import urlparse, parse_qs
import requests

#multi-threading
import concurrent.futures 

class alph_settings: 

    def __init__(self, apikey, site, date_range): 
        self.apikey = str(apikey) 
        self.site = str(site) 
        self.date_range = list(date_range)

class val_steps(alph_settings): 

    def get_tkers(self, function = "LISTING_STATUS", date_range = None):
        if date_range == None: 
            date_lst = sorted(self.date_range)
        else: 
            date_lst = sorted(date_range)
        #get range of tickers from dates provided
        lst_dict = {}
        for i in date_lst: 
            if i.month < 10 and i.day < 10:
                date_val = f"{i.year}-0{i.month}-0{i.day}"
            elif i.month < 10: 
                date_val = f"{i.year}-0{i.month}-{i.day}"
            elif i.day < 10:
                date_val = f"{i.year}-{i.month}-0{i.day}"
            else: 
                date_val = f"{i.year}-{i.month}-{i.day}"
            r = requests.get(f"{self.site}function={function}&date={date_val}&apikey={self.apikey}")
            decode_content = r.content.decode("utf-8")
            content = csv.reader(decode_content.splitlines(), delimiter = ",")
            lst_dict[i] = list(content)
        return lst_dict
    
    def get_valid_tkers(lst_dict, pretty_print = False): 
        key_lst = [i for i in lst_dict.keys()]
        len_lst = [len(lst_dict[i]) for i in key_lst]
        #logic to check shorter lists, can take more than two lists
        if statistics.mean(len_lst) < 1000: 
            compare_lst = lst_dict[key_lst[len_lst.index(max(len_lst))]]
            maintain_lst = []
            for i in compare_lst:
                for j, jalue in enumerate(key_lst): 
                    a_chk = []
                    if i in lst_dict[jalue]: 
                        a_chk = True
                    else: 
                        a_chk.append(jalue)
                if a_chk == True: 
                    maintain_lst.append(i)
                else: 
                    print(f"{i} was delisted in {min(a_chk)}")
        #write logic for longer lists but assume two dict keys only 
        else:
            for i in key_lst: 
                tker_mat = [lst_dict[i] for i in key_lst]
            mat1=set(tuple(x) for x in tker_mat[0])
            mat2=set(tuple(x) for x in tker_mat[1])
            maintain=set.intersection(mat1,mat2)
            delist=mat1^mat2
            maintain_lst=list(list(y) for y in maintain)
            if pretty_print == True:
                for j in delist:
                    print(f"Excluded: {j[0]}, {j[1]}")
            else: 
                pass
        return maintain_lst

class alph_api_wrapper(alph_settings): 

    def __init__(self, ticker_lst, slices): 
        self.ticker_lst = list(ticker_lst)
        self.interval = list(slices)

    def get_search(self, kwd_lst, function = "SYMBOL_SEARCH"):
        url_dict = {i: f"{self.site}function={function}&keywords={i}&apikey={self.apikey}" for i in kwd_lst}
        raw_response = []
        res_dict = {}
        for j in url_dict.keys(): 
            r = requests.get(url_dict[j])
            req = r.json()
            raw_response.append(req)
            res_dict[j] = [req["bestMatches"][k]["1. symbol"] for k in range(len(req["bestMatches"]))]
        return url_dict, raw_response, res_dict

    def intraday_url(self, function, ticker_lst, interval, slice = [], outputsize = "full", datatype = "csv", output = "json", adjusted = False): 
        if len(slice) == 0: 
            if output == "json": 
                url_dict = {i: f"{self.site}function={function}&symbol={i}&interval={interval}&outputsize={outputsize}&adjusted={adjusted}&datatype={datatype}&apikey={self.apikey}" for i in ticker_lst}
                return url_dict
            elif output == "lst": 
                url_lst = [f"{self.site}function={function}&symbol={i}&interval={interval}&outputsize={outputsize}&adjusted={adjusted}&datatype={datatype}&apikey={self.apikey}" for i in ticker_lst]
                return url_lst
        elif len(slice) == 1: 
            print("here")
            if output == "json": 
                url_dict = {i: f"{self.site}function={function}&symbol={i}&slice={slice}&interval={interval}&adjusted={adjusted}&outputsize={outputsize}&datatype={datatype}&apikey={self.apikey}" for i in ticker_lst}
                return url_dict
            elif output == "lst": 
                url_lst = [f"{self.site}function={function}&symbol={i}&slice={slice}&interval={interval}&adjusted={adjusted}&outputsize={outputsize}&datatype={datatype}&apikey={self.apikey}" for i in ticker_lst]
        elif len(slice) > 1:
            if output == "json":
                url_dict = {i: [f"{self.site}function={function}&symbol={i}&interval={interval}&slice={jalue}&outputsize={outputsize}&adjusted={adjusted}&datatype={datatype}&apikey={self.apikey}" for j, jalue in enumerate(slice)] for i in ticker_lst}
                return url_dict
            elif output == "lst": 
                url_lst = []
                for i in ticker_lst: 
                    for j in slice: 
                        url_lst.append(f"{self.site}function={function}&symbol={i}&interval={interval}&slice={j}&outputsize={outputsize}&adjusted={adjusted}&datatype={datatype}&apikey={self.apikey}")
                return(url_lst)
                #raise ValueError("Sorry, list output is unavaliable for multiple slices.")

    def get_general_url(self, function, ticker_lst, datatype, output = "json"): 
        if output == "json":
            url_dict = {i: f"{self.site}function={function}&outputsize=full&datatype={datatype}&symbol={i}&apikey={self.apikey}" for i in ticker_lst}
            return url_dict
        elif output == "lst": 
            url_lst = [f"{self.site}function={function}&outputsize=full&datatype={datatype}&symbol={i}&apikey={self.apikey}" for i in ticker_lst]
            return url_lst

    def get_json_data(url, keys = None, pretty_print = False): 
        r = requests.get(url)
        data = r.json()
        if keys == None: 
            keys = [i for i in data.keys()]
        else: 
            pass
        info_lst = [data.get(i) for i in keys]
        return info_lst

    def get_csv_data(url, pretty_print = False): 
        with requests.Session() as session: 
            load = session.get(url)
            decode = load.content.decode("utf-8")
            in_mem = csv.reader(decode.splitlines(), delimiter = ",")
            lst_data = list(in_mem)
            if pretty_print == False: 
                pass 
            elif pretty_print == True: 
                for row in lst_data[:50]: 
                    print(row)
            elif type(pretty_print) == int: 
                for row in lst_data[:pretty_print]: 
                    print(row)
            else: 
                print("Either bool or int, thanks. Skipping the print.")
                pass 
        return lst_data

class threading(alph_api_wrapper): 

    def thrd_ticker_slices(urls, wrapper_function = alph_api_wrapper.get_csv_data): 
        s = time.time()
        data = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(wrapper_function, url = url) for url in urls]
        for future in concurrent.futures.as_completed(futures):
            try:
                data.append(future.result()[1:])
            except Exception as exc:
                print('%r generated an exception: %s' % (exc))
            else:
                pass
        e = time.time()
        time_it = e-s
        print(f'Total time elapsed: {time_it} seconds')
        data = list(itertools.chain.from_iterable(data))
        return data

    def thrd_csv_data(urls, wrapper_function = alph_api_wrapper.get_csv_data, slice = None): 
        s = time.time()
        data = {}
        hit_again = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if slice == None:
                future_to_url = {executor.submit(wrapper_function, url = url): parse_qs(urlparse(url).query).get("symbol")[0] for url in urls}
            else: 
                future_to_url = {executor.submit(wrapper_function, url = url): slice + "_" + parse_qs(urlparse(url).query).get("symbol")[0] for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    if len(future.result()) < 5: 
                        hit_again.append(url)
                    else: 
                        data[url] = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (url, exc))
                else:
                    pass
        e = time.time()
        time_it = e-s
        print(f'Total time elapsed: {e-s} seconds')
        return hit_again, time_it, data

    def thrd_json_data(urls, wrapper_function, key_lst): 
        s = time.time()
        data = {}
        hit_again = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(wrapper_function, url = url, keys = key_lst): parse_qs(urlparse(url).query).get("symbol")[0] for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    if None in future.result(): 
                        hit_again.append(url)
                    else: 
                        data[url] = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (url, exc))
                else:
                    pass
        e = time.time()
        timeit = e-s
        print(f'Total time elapsed: {e-s} seconds')
        return hit_again, timeit, data
    
class filters(alph_settings): 

    def daily_filter(self, daily_data, pandas = True, ts_idx = 0, close_idx = 5, vol_idx = 6, header_row = 0): 
        start = self.date_range[0] 
        end = self.date_range[1]
        len_chk = []
        data_dict = {i: [[j[0], float(j[5]), int(j[6])] for j in daily_data[i][header_row+1:]] for i in daily_data.keys()}
        for i in data_dict.keys(): 
            data_dict[i] = pd.DataFrame(data_dict[i][header_row+1:], columns = [daily_data[i][header_row][0], daily_data[i][header_row][5], daily_data[i][header_row][6]])
            data_dict[i]["timestamp"] = pd.to_datetime(data_dict[i]["timestamp"])
            data_dict[i] = data_dict[i].query("@start <= timestamp <= @end")
            len_chk.append(len(data_dict[i]))
        if sum(len_chk)/len(len_chk) == len_chk[0]: 
            pass
        else:
            day_except = filters.debug_days(data_dict)
            for i in data_dict.keys(): 
                data_dict[i] = data_dict[i].query("timestamp != @day_except")
                data_dict[i] = data_dict[i].reset_index()
                if pandas == False:
                    data_dict[i] = data_dict[i].values.tolist()
                else: 
                    pass
        return data_dict, day_except

    def debug_days(data_dict, set_index = "timestamp"): 
        for i in data_dict.keys(): 
            #print(data_dict[i].columns)
            data_dict[i] = data_dict[i].set_index(set_index)
        debug_df = pd.concat([data_dict[i] for i in data_dict.keys()], axis = 1)
        debug_df = debug_df[debug_df.isnull().any(axis = 1)]
        debug_days = debug_df.reset_index()
        debug_days = debug_days["timestamp"].tolist()
        return debug_days

    def cofi_filter(data): 
        #clean out bad tickers
        bad_lst = [i for i in data.keys() if "None" in data[i]]
        filtered_cofi = {i: data[i] for i in data.keys() if "None" not in data[i] and "-" not in data[i]}
        return bad_lst, filtered_cofi

class make_dfs(): 

    def pivot_data(data_dict, key, index = 0): 
        pca_lst = []
        for i in data_dict.keys(): 
            pca_lst.append([i] + data_dict[i][key].tolist())
        pca_df = pd.DataFrame(data = pca_lst)
        pca_df = pca_df.set_index(index)
        pca_df = pca_df.astype(float)
        return pca_df

    def pivot_cofi(data_dict, index = 0): 
        pca_lst = []
        for i in data_dict.keys(): 
            pca_lst.append([i] + data_dict[i])
        pca_df = pd.DataFrame(data = pca_lst)
        pca_df = pca_df.set_index(index)
        pca_df = pca_df.astype(float)
        return pca_df
    
class helper(): 

    def reader_paths(folder = "/reader/", filetype = "*.csv", name = "", internal = True): 
        if internal:
            path = f"{os.path.dirname(__file__)}{folder}"
            reader_path = path+filetype
        else:
            cwd = os.path.abspath(os.getcwd())
            reader_path = cwd + folder + filetype
        files = glob.glob(reader_path)
        files = [i for i in files if name in i]
        latest = max(files, key = os.path.getctime)
        return files, latest
    
    def writer_path(folder = "/reader/"): 
        path = f"{os.path.dirname(__file__)}{folder}"
        return path
    
    def get_params(daily_data):
        data_dict = {}
        for j in daily_data:
            data_dict[j] = []
            for i in daily_data[j]:
                try:
                    data_dict[j].append([i[0], i[5], i[6]])
                except:
                    pass
        return data_dict
        
def test(): 
    print("hello")
