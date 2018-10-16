import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import quandl


stocks = ['AAPL']
m = 'PG'
quandl.ApiConfig.api_key = 'asLyyb8z8JXQ35tC89RF'
data = quandl.get_table('WIKI/PRICES', ticker = stocks,
                                       qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                                       date = { 'gte': '2013-1-1', 'lte': '2018-09-31' }, paginate=True)

data

df = data.set_index('date')
table = df.pivot(columns = 'ticker')
table.columns = [col[1] for col in table.columns]
table.head()

table['100 ma'] = table['AAPL'].rolling(window = 100, min_periods= 0).mean()
table.plot()
plt.show()
