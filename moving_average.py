import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
%matplotlib inline
import seaborn as sns
sns.set(style = 'darkgrid', context = 'talk', palette ='Dark2')
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf

my_year_month_fmt = mdates.DateFormatter('%m/%y')

tickers = ['AAPL', 'MSFT', '^GSPC']

start_date = '2010-01-01'
end_date = '2016-12-31'

yf.pdr_override()

panel_data = pdr.get_data_yahoo('AAPL', start_date, end_date)

panel_data

aapl = panel_data['Adj Close']

short_rolling_aapl = aapl.rolling(window =20).mean()
long_rolling_aapl = aapl.rolling(window = 100).mean()

fig, ax = plt.subplots(figsize = (16,9))
ax.plot(aapl.index, aapl, label = 'AAPL' )
ax.plot(short_rolling_aapl.index, short_rolling_aapl, label = '20 day rolling')
ax.plot(long_rolling_aapl.index, long_rolling_aapl, label = '100 day rolling')
ax.set_xlabel('Date')
ax.set_ylabel('Adjust closing price ($)')
ax.legend()
