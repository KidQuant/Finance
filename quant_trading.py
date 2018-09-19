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
