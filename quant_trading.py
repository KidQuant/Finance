import pandas as pd
import numpy as np
import patsy

pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data
import quandl
quandl.ApiConfig.api_key = 'E5YTXWQoe7sxQh1ZCEzt'

df = quandl.get('WIKI/' + 'AAPL', start_date= '2013-01-01')

len(df)
df['Split Ratio'].value_counts()
df[df['Split Ratio'] == 7.0]

aapl_split = quandl.get('WIKI/' + 'AAPL', start_date = '2014-06-10')
aapl_split.head()

import statsmodels.tsa.stattools as ts
cadf = ts.adfuller(aapl_split.Close)

print('Augment Dickey Fuller')
print('Test Statistic =',cadf[0])
print('p-Value =',cadf[1])
print('Criical Values =', cadf[4])
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

MSFT = quandl.get('WIKI/' + 'MSFT', start_date='2014-06-10')
INTC = quandl.get('WIKI/' + 'INTC', start_date='2014-06-10')
TIF = quandl.get('WIKI/' + 'TIF', start_date='2014-06-10')

sns.jointplot(INTC.Close, aapl_split.Close, kind='reg');

np.corrcoef(INTC.Close, aapl_split.Close)

np.corrcoef(INTC.Close[:-7], aapl_split.Close[7:])

#Google Trends
aapl_trends = pd.read_csv('multiTimeLine.csv', header=1)

aapl_trends.tail()

aapl_split_week = aapl_split.resample('W', convention = 'end').last()

#trend and price corr

np.corrcoef(aapl_trends['Apple: (Worldwide)'],aapl_split_week.Close)
