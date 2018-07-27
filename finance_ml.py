import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import bs4 as bs
import requests
import pickle
import os
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web

#Getting a list from S&P 500 companies from Wikipedia.

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
#This eliminates all of the tickers the get_data_from_morningstar function has a difficult time extracting

        if ticker == 'ANDV':
            next
        elif ticker == 'BKNG':
            next
        elif ticker == 'BHF':
            next
        elif ticker == 'CBRE':
            next
        elif ticker == 'DWDP':
            next
        elif ticker == 'DXC':
            next
        elif ticker == 'EVRG':
            next
        elif ticker == 'JEF':
            next
        elif ticker == 'TPR':
            next
        elif ticker == 'UAA':
            next
        elif ticker == 'WELL':
            next
        else:
            tickers.append(ticker)

    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)

    return tickers

save_sp500_tickers()

#Retracting all of the S&P 500 tickers, with a few expectations

def get_data_from_morningstar(reload_sp500=False):
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
            df = web.DataReader(ticker, 'morningstar', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df = df.drop("Symbol", axis=1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

get_data_from_morningstar()

#Compiling all of the closing prices into a single dataframe and csv

def compile_data():
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Close':ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

        print(main_df.head())
        main_df.to_csv('sp500_joined_closes.csv')

compile_data()
