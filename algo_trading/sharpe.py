
#sharpe.py

from __future__ import print_function

import datetime
import numpy as np
import pandas as pd
from pandas_datareader import data as web

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

    start = datetime.datetime(2017,1,1)
    end = datetime.datetime.now()

    
