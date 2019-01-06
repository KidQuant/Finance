
# cadf.py

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pandas_datareader import data as pdr
import pprint
import statsmodels.tsa.stattools as ts
import fix_yahoo_finance as yf

from pyfinance.ols import PandasRollingOLS

yf.pdr_override()

start = dt.datetime(2016,1,1)
end = dt.datetime(2018,1,1)

wll = pdr.get_data_yahoo('WLL', start, end)
wll


def plot_price_series(df, ts1, ts2):
    global start, end
    months = mdates.MonthLocator() #every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df[ts1], label=ts1)
    ax.plot(df.index, df[ts2], label=ts2)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(start, end)
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('%s and %s Daily Prices' % (ts1,ts2))
    plt.legend()
    plt.show()

def plot_scatter_series(df, ts1, ts2):
    plt.xlabel('%s Price ($)' % ts1)
    plt.ylabel('%s Price ($)' % ts2)
    plt.title('%s and %s Price Scatterplot' % (ts1, ts2))
    plt.scatter(df[ts1], df[ts2])
    plt.show()

def plot_residuals(df):
    global start, end
    start = dt.datetime(2017,1,1)
    end = dt.datetime(2018,1,1)
    months = mdates.MonthLocator()
    fig, ax = plt.subplot()
    ax.plot(df.index, df['res'], label = 'Residuals')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(start, end)
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('Residual Plot')
    plt.legend()

    plt.plot(df['res'])
    plt.show()

if __name__ == '__main__':

    start = dt.datetime(2016,1,1)
    end = dt.datetime(2018,1,1)

    ar = pdr.get_data_yahoo('AR', start, end)
    wll = pdr.get_data_yahoo('WLL', start, end)

    df = pd.DataFrame(index = ar.index)
    df['AR'] = ar['Adj Close']
    df['WLL'] = wll['Adj Close']

    # Plot the two time series
    plot_price_series(df, 'AR', 'WLL')

    #Display a scatter plot of the two time series
    plot_scatter_series(df, 'AR', 'WLL')

    #Calculate optimal hedge ratio 'beta'
    res = PandasRollingOLS(y = df['WLL'], x = df['AR'])
    beta_hr = res.beta.x

    df['res'] = df['WLL'] - beta_hr * df['AR']

    plot_residuals(df)

    cadf = ts.adfuller(df['res'])
    pprint.pprint(cadf)
