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
                                       date = { 'gte': '2016-1-1', 'lte': '2018-09-31' }, paginate=True)

data

df = data.set_index('date')
table = df.pivot(columns = 'ticker')
table.columns = [col[1] for col in table.columns]
table.head()

table['mu'] = [table[m][:i].mean() for i in range(len(table))]
plt.figure(figsize=(15,7))
plt.plot(table)
plt.plot(table['mu'])
plt.show()
