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

panel_data = pdr.get_data_yahoo('INPX', start_date, end_date)

panel_data

inpx = panel_data['Adj Close']

short_rolling_inpx = inpx.rolling(window =20).mean()
long_rolling_inpx = inpx.
