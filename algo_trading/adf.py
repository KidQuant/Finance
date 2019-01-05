from __future__ import print_function
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import statsmodels.tsa.stattools as ts
import datetime as dt

yf.pdr_override()

start = dt.datetime(2005,1,1)
end = dt.datetime.now()

amzn = pdr.get_data_yahoo('AMZN', start, end)['Adj Close']

ts.adfuller(amzn)
