import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from pandas_datareader import data as web

style.use('ggplot')

start = dt.datetime(2017,1,1)
end = dt.datetime.now()

df = web.DataReader('TSLA', 'robinhood', start, end)

print(df.head())

df.reset_index(inplace=True)
df.set_index('begins_at', inplace=True)
df = df.drop(['symbol', 'interpolated', 'session'], axis=1)

print(df.head())

df.to_csv('TSLA.csv')
df = pd.read_csv('TSLA.csv', parse_dates = True, index_col = 0)

df.plot()
plt.show()
plt.savefig('TSLA.jpeg', bbox_inches ='tight')

df['close_price'].plot()
plt.show()
plt.savefig('TSLA1.jpeg', bbox_inches = 'tight')

df['20ma'] = df['close_price'].rolling(window=20).mean()
print(df.head())

df['20ma'] = df['close_price'].rolling(window = 20, min_periods =0).mean()
print(df.head())

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=5, colspan=1, sharex=ax1)
ax1.plot(df.index, df['close_price'])
ax1.plot(df.index, df['20ma'])
ax2.bar(df.index, df['volume'])
plt.savefig('TSLAmoving', bbox_inches='tight')
plt.show()

import bs4 as bs
import pickle
import requests

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

import os

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
