
#sharpe.py

from __future__ import print_function

import datetime
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import fix_yahoo_finance

def annualized_sharpe(returns, N=252):
    """
    Calculate the annualized Sharpe ratio of a return stream
    based on a number of trading periods, N. N defaults to 252,
    which then assumes a stream of daily return.

    The function assumes that the returns are the excess of
    those compared to a benchmark.
    """

    return np.sqrt(N) * returns.mean() / returns.std()

def equity_sharpe(ticker):
    """
    Calulates the annualized Sharpe ratio based on the daily
    returns of an equity ticker symbol listed in RobinHood.

    The dates have been hardcoded here for brevity.
    """

    start = datetime.datetime(2013,1,1)
    end = datetime.datetime.now()

    # Obtain the equities daily historic data for the desired time period
    # and add to pandas DataFrame
    pdf = web.get_data_yahoo(ticker, start, end)

    # Use the percentage change method to calculate daily returns
    pdf['daily_ret'] = pdf['Adj Close'].pct_change()

    # Assume an average annual risk-free rate over the period of 5%
    pdf['excess_daily_ret'] = pdf['daily_ret'] - 0.05 / 252

    # Returns the annualized Sharpe ratio based on the excess daily returns
    return annualized_sharpe(pdf['excess_daily_ret'])

equity_sharpe('GOOGL')

def market_neutral_sharpe(ticker, benchmark):
    """
    Calculates the annualized Sharpe ratio of a market
    neutral long/short strategy involving the long of 'ticker'
    with a corresponding short of the 'benchmark'.
    """

    start = datetime.datetime(2013,1,1)
    end = datetime.datetime.now()

    # Get historic data for both a symbol/ticker and a benchmark ticker
    # The dates have been hardcoded, but you can modify them as you see fit!
    tick = web.get_data_yahoo(ticker, start, end)
    bench = web.get_data_yahoo(benchmark, start, end)

    # Calculate the percentage returns on each of the time series
    tick['daily_ret'] = tick['Adj Close'].pct_change()
    bench['daily_ret'] = bench['Adj Close'].pct_change()

    # Create a new DataFrame to store the strategy information
    # The net returns are (long- short)/2, since there is twice
    # the trading capital for this strategy
    strat = pd.DataFrame(index = tick.index)
    strat['net_ret'] = (tick['daily_ret'] - bench['daily_ret'])/2.0

    #Return the annualized Sharpe ratio for this strategy
    return annualized_sharpe(strat['net_ret'])

market_neutral_sharpe('GOOGL', 'SPY')


web.get_data_yahoo('AAPL', '2015-01-01', '2018-07-01')
