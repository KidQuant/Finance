import datetime as dt
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt

yf.pdr_override()

#Download data from yahoo finance
start = dt.datetime(2009,1,1)
end = dt.datetime.now()

df = pdr.get_data_yahoo('AAPL', start, end)
df['Adj Close'].plot()
plt.title("AAPL Price History")
plt.show()