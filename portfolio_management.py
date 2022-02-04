import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as web
from functools import reduce
from tabulate import tabulate
from matplotlib.ticker import FormatStrFormatter

def display(data):
	print(tabulate(data, headers = 'keys', tablefmt = 'psql'))
	return

# 1 - Define `tickers` & `company names` for every instrument
stocks      = {'AAPL':'Apple', 'MSFT':'Microsoft', 'AMZN' : 'Amazon',  'GOOG': 'Google', 'FB':'Facebook','NFLX':'Netflix' ,  'NVDA' : 'NVIDIA'}
bonds       = {'HCA' : 'HCA', 'VRTX' :  'VRTX'}
commodities = {'BTC-USD' : 'Bitcoin', 'PA=F' : 'Palladium'}
instruments = {**stocks, **bonds, **commodities}
tickers     = list(instruments.keys())
instruments_data = {}
N = len(tickers)

# 2 - We will look at stock prices over the past years, starting at January 1, 2015
# 01-01-2015 - 16-04-2020
start = datetime.datetime(2015,1,1)
end = datetime.datetime(2020, 4, 16)

# 3 - Let's get instruments data based on the tickers.
# First argument is the series we want, second is the source ("yahoo" for Yahoo! Finance), third is the start date, fourth is the end date
for ticker, instrument in instruments.items():
    print("Loading data series for instrument {} with ticker = {}".format(instruments[ticker], ticker))
    instruments_data[ticker] = web.DataReader(ticker, data_source = 'yahoo', start=start, end = end)

instruments_data['AAPL']

# 2.3.1 - keep only 'adjusted close' prices
for ticker, instrument in instruments.items():
    instruments_data[ticker] = instruments_data[ticker]["Adj Close"]

# 2.3.2 - Drop duplicates for Palladium data from Yahoo Source
instruments_data['PA=F'] = instruments_data['PA=F'].drop_duplicates()

tr_days = [ len(instr) for _, instr in instruments_data.items() ]
tr_days = pd.DataFrame(tr_days, index = tickers, columns = ["Trading Days"])

tr_days.T

tr_days_stocks_bonds = instruments_data['AAPL'].groupby([instruments_data['AAPL'].index.year]).agg('count')
tr_days_bitcoin = instruments_data['BTC-USD'].groupby([instruments_data['BTC-USD'].index.year]).agg('count')
tr_days_palladium = instruments_data['PA=F'].groupby([instruments_data['PA=F'].index.year]).agg('count')

tr_days_per_year = pd.DataFrame([tr_days_stocks_bonds, tr_days_bitcoin, tr_days_palladium], index=["Stocks & Bonds", "Bitcoin", "Palladium"])

tr_days_per_year

## 2.4 - Merging Dataframes
'''
    instruments_data = {'AAPL' : dataframe (1331 x 1),..., 'BTC-USD' : dataframe (1934 x 1), 'PA=F' : dataframe (1336 x 1)}
    [*] So list(instruments_data.values()) : we only keep the dataframes in a list
    [*] data_df = pd.concat(data, axis = 1).dropna() DID not wor because of different `commodities` sizes

'''

data = list(instruments_data.values())
data_df = reduce(lambda x, y: pd.merge(x, y, left_index = True, right_index = True, how = 'outer'), data).dropna()
data_df.columns = tickers

data_df

tr_days_per_year = data_df['AAPL'].groupby([data_df['AAPL'].index.year]).agg('count')
tr_days_per_year = pd.DataFrame([tr_days_per_year], index = ["All instruments (merged)"])

tr_days_per_year

fig, ax = plt.subplots(figsize=(12, 8))
data_df.plot(ax = plt.gca(), grid = True)
ax.set_title('Adjusted Close for all instruments')
ax.set_facecolor((0.95, 0.95, 0.99))
ax.grid(c = (0.75, 0.75, 0.99))

simple_returns = data_df.apply(lambda x: x /x[0] - 1)
simple_returns.plot(grid = True, figsize = (10, 5)).axhline(y = 0, color = "black", lw=2)

log_returns = data_df.pct_change()
log_returns

log_returns.plot(grid = True, figsize = (15,10)).axhline(y = 0, color = "black", lw=2)

APR = log_returns.groupby([log_returns.index.year]).agg('sum')
APR_avg = APR.mean()

APR

pd.DataFrame(APR_avg, columns = ['Average APR']).T

N = np.array(tr_days_per_year.T)
N_total = np.sum(N)
APY = (1 + APR / N)**N-1
APY_avg = (1 + APR_avg/N_total)**N_total - 1

APY

pd.DataFrame(APY_avg, columns = ['Average APY']).T


STD = log_returns.groupby([log_returns.index.year]).agg('std') * np.sqrt(N)
STD_avg = STD.mean()
std = log_returns.std()

STD

pd.DataFrame(STD_avg, columns = ['Average STD']).T

# configuration
fig, ax = plt.subplots(figsize = (16,12))
ax.set_title(r"Standard Deviation ($\sigma$) of all instruments for all years")
ax.set_facecolor((0.95, 0.95, 0.99))
ax.grid(c = (0.75, 0.75, 0.99))
ax.set_ylabel(r"Standard Deviation $\sigma$")
ax.set_xlabel(r"Years")
STD.plot(ax = plt.gca(),grid = True)

for instr in STD:
  stds = STD[instr]
  years = list(STD.index)
  for year, std in zip(years, stds):
    label = "%.3f"%std
    plt.annotate(label, xy = (year, std), xytext=((-1)*50, 40),textcoords = 'offset points', ha = 'right', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
      arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
