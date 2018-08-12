import matplotlib.pyplot as plt
import pandas_datareader.data as web
import numpy as np

# collect data for Amazon from 2017-04-22 to 2018-04-22
start = '2017-04-22'
end = '2018-04-22'
symbol = 'AMZN'
df = web.DataReader(name= symbol, data_source='iex', start=start, end=end)
print(df)
df.to_csv("Risk_Management/{}.csv".format(symbol))

close = df[['close']]
close = close.rename(columns={'close': symbol})
ax = close.plot(title='Amazon')
ax.set_xlabel('date')
ax.set_ylabel('close price')
ax.grid()
plt.show()

periods = 252
noise = np.random.rand(252)
rng = pd.date_range('1/1/2011', periods = periods, freq='D')
stk1 = (np.arange(1,1+(.001)*(periods), .001)) * 30 + noise
stk2 = (np.arange(1,1+(.001)*(periods), .001)) * 30 - noise

portfolio = .5 * stk1 + .5 * stk2
df = pd.DataFrame(index=rng, data = {'STK1': stk1, 'STK2': stk2, 'PORTFOLIO':portfolio})
print(df.head())

ax = df.plot(title='Stock Price')
ax.set_xlabel('data')
ax.set_ylabel('close price')
ax.grid()
plt.show()
