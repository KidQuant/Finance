import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
from pandas_datareader import data as pdr
from matplotlib import style
import pandas as pd
import fix_yahoo_finance as yf
import numpy as np

yf.pdr_override()

import bs4 as bs
import pickle
import requests
import os

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('.','-')
        tickers.append(ticker)

    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)

    return tickers

save_sp500_tickers()

def get_data_from_yahoo(reload_sp500=False):

    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2013, 1, 1)
    end = dt.datetime(2018,1,1)
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            try:
                df = pdr.get_data_yahoo(ticker, start, end)
                df.reset_index(inplace=True)
                df.set_index('Date', inplace=True)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            except ValueError:
                pass
        else:
            print('Already have {}'.format(ticker))

get_data_from_yahoo(reload_sp500=True)
