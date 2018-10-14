import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import bs4 as bs
import requests
import pickle
import os
pd.core.common.is_list_like = pd.api.types.is_list_like
from  pandas_datareader import data as web
from collections import Counter
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn import neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import quandl

quandl.ApiConfig.api_key = 'asLyyb8z8JXQ35tC89RF'


#Getting a list from S&P 500 companies from Wikipedia.

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)

    return tickers

save_sp500_tickers()

#Retracting all of the S&P 500 tickers, with a few expectations

def get_data_from_robinhood(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'robinhood', start, end)
            df.reset_index(inplace=True)
            df.set_index('begins_at', inplace=True)
            df = df.drop('symbol', axis=1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

get_data_from_robinhood()

def compile_data():
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('begins_at', inplace=True)

        df.rename(columns={'close_price':ticker}, inplace=True)
        df.drop(['high_price', 'interpolated', 'low_price', 'open_price','session', 'volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

        print(main_df.head())
        main_df.to_csv('sp500_joined_closes.csv')

compile_data()

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

process_data_for_labels('XOM')

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirements = 0.02
    for col in cols:
        if col > requirements:
            return 1
        if col < -requirements:
            return -1
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    return X, y, df

extract_featuresets('XOM')

def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.25)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))
    print()
    print()
    return confidence

do_ml('XOM')

from statistics import mean

with open('sp500tickers.pickle', 'rb') as f:
    tickers = pickle.load(f)

accuracies = []
for count, ticker in enumerate(tickers):

    if count% 10 == 0:
        print(count)

        accuracy = do_ml(ticker)
        accuracies.append(accuracy)
        print('{} accuracy: {}. Average accuracy:'.format(ticker, accuracy, mean(accuracies)))
