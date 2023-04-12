#The usual suspects
from itertools import combinations
import pandas as pd 
import numpy as np
import json
import copy
import time
from datetime import datetime 
import re
import pandas as pd

#Dimensionality reduction and clustering dependencies
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

#ADF dependencies
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import coint

def apply_pca(price_df, num):
    pca = PCA(n_components=num)
    principalComponents = pca.fit_transform(price_df)
    explained_var = pca.explained_variance_ratio_.sum()
    principalDf = pd.DataFrame(data = principalComponents, columns = [f"price_component_{i}" for i in range(0, num)], index = price_df.index)
    return principalComponents, principalDf, explained_var

def apply_tsne(price_pca, perplexity = 15):
    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=300)
    tsne_pca_results = tsne.fit_transform(price_pca)
    tsne_df = pd.DataFrame(tsne_pca_results)
    return tsne_pca_results, tsne_df

def dbscan_results(tsne_pca_results, price_df):
    dbs=DBSCAN(eps=0.7,min_samples=2,n_jobs=1)
    print(dbs)
    #Noisy samples are given label -1
    dbs.fit(pd.DataFrame(tsne_pca_results))
    labels=dbs.labels_
    n_cl=len(set(labels)) - (1 if -1 in labels else 0)
    #print(f'{n_cl} clusters')
    cl_lab=dbs.labels_
    #cl_series creates a pd series with tickers as index and the cluster label as values
    cl_series=pd.Series(index=price_df.index,data=cl_lab.flatten())
    #cl_series_2 excludes cluster 1
    cl_series_2=cl_series[cl_series!=1]
    cnt=cl_series_2.value_counts()
    tick_cnt=cnt[(cnt>1)&(cnt<=100)]
    print(f"Clusters formed: {len(tick_cnt)}")
    cl_lst=list(cnt[(cnt < 500) & (cnt > 1)].index)[::-1]
    plt.figure(1, facecolor = 'white', figsize = (15, 10))
    plt.clf()
    plt.axis('off')
    plt.scatter(tsne_pca_results[(labels!=-1), 0], tsne_pca_results[(labels!=-1), 1], s = 100, alpha = 0.85, c = labels[labels!=-1], cmap = cm.Paired)
    plt.scatter(tsne_pca_results[(cl_series==-1).values, 0], tsne_pca_results[(cl_series==-1).values, 1])
    plt.figure(figsize = (12, 7))
    plt.barh(range(len(cl_series.value_counts())), cl_series.value_counts())
    return cl_series

def pairs_select(cl_series, raw_prices_df, labels, sf = 0.05):
    if labels == "all":
        df = pd.DataFrame(cl_series, columns = ["label"])
    else: 
        if type(labels) == int:
            df = pd.DataFrame(cl_series, columns = ["label"]).query("label == @labels")
        elif type(labels) == list:
            df = pd.DataFrame(cl_series, columns = ["label"]).query("label in @labels")
    p_vals = []
    pair_coint = []
    grouped_tickers = dict(df.groupby("label").groups)
    for i in grouped_tickers.values():
        tickers = list(i)
        stock_pairs = list(combinations(tickers, 2))
        for pair in stock_pairs:
            if pair[0] in raw_prices_df.columns and pair[1] in raw_prices_df.columns:
                results = sm.OLS(raw_prices_df[pair[0]].tolist(), raw_prices_df[pair[1]].tolist()).fit()
                predict = results.predict(raw_prices_df[pair[1]].tolist())
                error = raw_prices_df[pair[0]].to_list() - predict
                ADFtest = ts.adfuller(error)
                if ADFtest[1] < sf:
                    p_vals.append(ADFtest[1])
                    pair_coint.append(pair)
                    #print(ADFtest[1], pair)
    return p_vals, pair_coint

def get_tickers(p_vals, pair_coint, n = 5): 
    df = pd.DataFrame(data = {"p_value": p_vals, "pair": pair_coint})
    df.sort_values(by = "p_value", inplace = True)
    pairs = df["pair"].tolist()[:n]
    return pairs