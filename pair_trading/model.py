import pandas as pd
import numpy as np
import pandas_ta as pta

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit, RepeatedStratifiedKFold
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

import pickle

from pair_trading.preproc import helper

def get_features(ticker_pair):
    _, df1path = helper.reader_paths(name = ticker_pair[0])
    _, df2path = helper.reader_paths(name = ticker_pair[1])
    df1 = pd.read_csv(df1path)
    df2 = pd.read_csv(df2path)
    df1.set_index('time', inplace=True)
    df1.sort_index(inplace=True)
    df2.set_index('time', inplace=True)
    df2.sort_index(inplace=True)
    #VWAP
    def vwap(df):
        df['VWAP'] = (df.volume*(df.high+df.low)/2).cumsum() / df.volume.cumsum()
        return df
    #SMA
    def sma(df, num):
        df[f"SMA({num})"] = df.close.rolling(num).mean()
        return df
    #EWM
    def ewm(df):
        df['12dayEWM'] = df.close.ewm(span=5, adjust=False).mean()
        return df
    #MACD
    def macd(df):
        df['MACD'] = pta.macd(df['close']).iloc[:,0]
        return df
    #RSI
    def rsi(df):
        df["RSI"] = pta.rsi(df['close'], length = 12)
        return df
    #MOM
    def mom(df):
        df["rolling"] = df.close.shift(12)
        df["MOM"] = df["close"] - df["rolling"]
        df.drop(["rolling"], axis=1, inplace=True)
        return df
    #MFI
    def mfi_calc(h, l, o, c, v, n=12):
        typical_price = (h+l+c)/3
        money_flow = typical_price*v
        mf_sign = np.where(typical_price > typical_price.shift(1),1,-1)
        signed_mf = money_flow * mf_sign
        mf_ave_gain = signed_mf.rolling(n).apply(lambda x: ((x>0)*x).sum(), raw = True)
        mf_ave_loss = signed_mf.rolling(n).apply(lambda x: ((x<0)*x).sum(), raw = True)
        return (100-(100/(1+mf_ave_gain / abs(mf_ave_loss)))).to_numpy()
    def mfi(df):
        df["MFI"] = mfi_calc(df.high,df.low,df.open,df.close,df.volume)
        return df
    def run_all(df):
        df = vwap(df)
        df = sma(df, 5)
        df = sma(df, 10)
        df = ewm(df)
        df = macd(df)
        df = rsi(df)
        df = mom(df)
        df = mfi(df)
        return df
    df1_features = run_all(df1)
    df2_features = run_all(df2)
    ratio_features = ['VWAP','SMA(5)','SMA(10)','12dayEWM','RSI']
    diff_features = ['MACD','MOM','MFI']
    ratio_features_df = df1_features[ratio_features]/df2_features[ratio_features]
    diff_features_df = df1_features[diff_features] - df2_features[diff_features]
    diff_features_df['spread'] = df1['close'] - df2['close']
    features_df = pd.concat([ratio_features_df, diff_features_df], axis=1).dropna()
    return features_df

def get_labels(df, t, threshold, summary = True):
    """
    df: df containing spread to identify buy, hold, or sell signal
    t: % return on spread t hours later
    threshold: threshold for labelling on the forward return
    Labels: 
    (1) --> BUY: If the return is more than x%, we should have bought (1)
    (2) --> SELL: If return is less than x%, we should have sold (-1)
    (3) --> HOLD: If in between, do nothing (0)
    """
    # Calculate % return on spread t hours later
    df['forward_return'] = df['spread'].diff(periods=t)/df['spread']
    #Set labels
    df['output'] = np.select([df['forward_return'] > threshold, df['forward_return'] < -threshold], [1,-1])
    if summary == True:
        print(
            f"""
            Count of Trades Identified:
            Buy signals: {len(df.query("output == 1")["output"].tolist())}
            Sell signals: {len(df.query("output == -1")["output"].tolist())}
            Hold signals: {len(df.query("output == 0")["output"].tolist())}
            """)
    return df

def run_baseline(df, interval, thereshold):
    # reference https://blog.quantinsti.com/pairs-trading-basics/
    # 1 is go short (sell es buy wec), 0 is hold, -1 is go long 
    # spr["label"] = 1 if spr[spr["z-score"] > spr["upper_threshold"]] else 0 if spr[spr["z-score"] < spr["lower_threshold"]] else -1
    df["rolling_mean"] = df['spread'].rolling(interval).mean()
    df["rolling_std"] = df['spread'].rolling(interval).std()
    df.dropna(inplace=True)
    df["upper_threshold"] = thereshold*df["rolling_std"]
    df["lower_threshold"] = - thereshold*df["rolling_std"]
    df["z-score"] = (df['spread'] - df["rolling_mean"]) / df["rolling_std"]
    df['label'] = df.apply(lambda x: 1 if x["z-score"] > x["upper_threshold"] else 0 if x["z-score"] < x["lower_threshold"] else -1, axis=1)
    split = round(0.8*len(df))
    train, test = df[:split],df[split:]
    x_train = train[['VWAP','SMA(5)','SMA(10)','12dayEWM','RSI','MACD','MOM','MFI','spread']]
    y_train = train[['output']]
    x_test = test[['VWAP','SMA(5)','SMA(10)','12dayEWM','RSI','MACD','MOM','MFI','spread']]
    y_test = test[['output']]
    y_pred = test[['label']]
    print(classification_report(y_test,y_pred))
    return y_pred

def run_svm(df, t = 24, thres = 0.05, split = 0.8, cv_summary = True):
    #Construct x and y
    X = df.loc[:,df.columns != 'output'].copy()
    X = (X-X.mean())/(X.max()-X.min())
    X['spread'] = df['spread']
    X.dropna(inplace = True)
    Y = get_labels(X, t, thres, summary = False)
    Y.dropna(inplace = True)
    Y = Y["output"].tolist()
    #train test split
    split = int(len(df) * split)
    X_train = X[:split]
    y_train = Y[:split]
    X_test = X[split:]
    y_test = Y[split:]
    #split = round(split*len(df))
    #train, test = df[:split],df[split:]
    #X_train = train[['VWAP','SMA(5)','SMA(10)','12dayEWM','RSI','MACD','MOM','MFI','spread']]
    #y_train = train[['output']]
    #X_test = test[['VWAP','SMA(5)','SMA(10)','12dayEWM','RSI','MACD','MOM','MFI','spread']]
    #y_test = test[['output']]
    #get cross validation score
    tscv = TimeSeriesSplit()
    clf = SVC(decision_function_shape='ovo',class_weight='balanced')
    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]},
                   {'kernel': ['linear'], 'C': [1, 10, 100]},
                   {'kernel':['poly'],'degree':[2,3,4],'C': [1, 10, 100],'gamma': ['auto']}
                   ]
    cross_val_score(clf, X_train, y_train, cv=tscv, scoring='f1_weighted')
    print(cross_val_score)
    finder = GridSearchCV(clf, params_grid, cv=tscv, scoring='f1_weighted')
    fin = finder.fit(X_train, y_train)
    print(fin)
    best_params = finder.best_params_
    best_score = finder.best_score_
    print(
        f"""
        Best Parameters: {best_params}
        Best Cross-Validation Score: {best_score}
        """)
    y_pred = finder.predict(X_test)
    print("\n")
    print(classification_report(y_test,y_pred))
    with open('svm_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    return y_pred, best_params

def run_logistic_regression(df, split = 0.8):
    split = round(split*len(df))
    #make train/test split
    train, test = df[:split],df[split:]
    x_train = train[['VWAP','SMA(5)','SMA(10)','12dayEWM','RSI','MACD','MOM','MFI','spread']]
    y_train = train[['output']]
    x_test = test[['VWAP','SMA(5)','SMA(10)','12dayEWM','RSI','MACD','MOM','MFI','spread']]
    y_test = test[['output']]
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    #log reg
    penalty = ['l1','l2', 'none', 'elasticnet']
    c_values = [1e-3, 1e-2, 1e-1, 1, 10]
    #define grid search
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    model = GridSearchCV(estimator=LogisticRegression(), param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = model.fit(x_train, y_train.values.ravel())
    #summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    y_pred = model.predict(x_test)
    print(classification_report(y_test,y_pred))
    best_params = model.best_params_
    with open('logreg_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    return y_pred, best_params

def run_random_forest(df, split = 0.8):
    split = round(split*len(df))
    #make train/test split
    train, test = df[:split],df[split:]
    x_train = train[['VWAP','SMA(5)','SMA(10)','12dayEWM','RSI','MACD','MOM','MFI','spread']]
    y_train = train[['output']]
    x_test = test[['VWAP','SMA(5)','SMA(10)','12dayEWM','RSI','MACD','MOM','MFI','spread']]
    y_test = test[['output']]
    param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 100, 120],
    'max_features': [9],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]}
    # Instantiate the grid search model
    rf = RandomForestClassifier()
    rf.fit(x_train,y_train)
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(x_train, y_train)
    grid_search.best_params_
    rf_model = RandomForestClassifier(bootstrap= True,
                             max_depth= 120,
                             max_features= 9,
                             min_samples_leaf= 5,
                             min_samples_split= 10,
                             n_estimators=100,
                             random_state = 42)
    rf_model.fit(x_train, y_train)
    y_pred_rf = rf_model.predict(x_test)
    best_params = grid_search.best_params_
    print(classification_report(y_test,y_pred_rf))
    with open('rf_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    return y_pred_rf, best_params

def make_models(svm_params, logreg_params, rf_params):
    #Parse best parameters for svc
    svm_c = svm_params["C"]
    svm_gamma = svm_params["gamma"]
    svm_kernel = svm_params["kernel"]
    svm_model = SVC(C=svm_c, 
                    degree= 2, 
                    gamma=svm_gamma, 
                    kernel=svm_kernel)
    #Parse best parameters for logistic regression
    lr_c = logreg_params["C"]
    lr_penalty = logreg_params["penalty"]
    lr_solver = logreg_params["solver"]
    logreg_model = LogisticRegression(solver=lr_solver,
                                      penalty=lr_penalty,
                                      C=lr_c, 
                                      max_iter=5000)
    #Parse best parameters for random forest
    rf_bs = rf_params["bootstrap"]
    rf_maxdepth = rf_params["max_depth"]
    rf_maxfeat = rf_params["max_features"]
    rf_minleaf = rf_params["min_samples_leaf"]
    rf_minsplit = rf_params["min_samples_split"]
    rf_nest = rf_params["n_estimators"]
    rf_model = RandomForestClassifier(bootstrap=rf_bs,
                                      max_depth= rf_maxdepth,
                                      max_features=rf_maxfeat,
                                      min_samples_leaf=rf_minleaf,
                                      min_samples_split=rf_minsplit,
                                      n_estimators=rf_nest,
                                      random_state=42)
    return svm_model, logreg_model, rf_model

def run_stacking(df, svm_model, logreg_model, rf_model, split = 0.8):
    split = round(split*len(df))
    #make train/test split
    train, test = df[:split],df[split:]
    x_train = train[['VWAP','SMA(5)','SMA(10)','12dayEWM','RSI','MACD','MOM','MFI','spread']]
    y_train = train[['output']]
    x_test = test[['VWAP','SMA(5)','SMA(10)','12dayEWM','RSI','MACD','MOM','MFI','spread']]
    y_test = test[['output']]
    lvl_0 = [("svm", svm_model), ("lr", logreg_model), ("rf", rf_model)]
    lvl_1 = LogisticRegression()
    model = StackingClassifier(estimators=lvl_0, 
                               final_estimator=lvl_1, 
                               cv=5)
    stacking_model = model.fit(x_train,y_train)
    stacking_y_pred = stacking_model.predict(x_test)
    print(classification_report(y_test, stacking_y_pred))
    print(accuracy_score(stacking_y_pred, y_test))
    return stacking_y_pred

def run_max_voting(df, svm_model, logreg_model, rf_model, split = 0.8):
    split = round(split*len(df))
    #make train/test split
    train, test = df[:split],df[split:]
    x_train = train[['VWAP','SMA(5)','SMA(10)','12dayEWM','RSI','MACD','MOM','MFI','spread']]
    y_train = train[['output']]
    x_test = test[['VWAP','SMA(5)','SMA(10)','12dayEWM','RSI','MACD','MOM','MFI','spread']]
    y_test = test[['output']]
    estimators = [("svm", svm_model), ("lr", logreg_model), ("rf", rf_model)]
    model = VotingClassifier(estimators=estimators, voting='hard')
    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_pred, y_test))
    return y_pred

def test():
    print("hello test")