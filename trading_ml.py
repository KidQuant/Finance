from pandas_datareader import data as web
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV as rcv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from IPython import get_ipython
import datetime as dt

start = dt.datetime(2017,1,1)
end = dt.datetime.now()


avg_err ={}
avg_train_err = {}
df = web.DataReader('SPY', 'robinhood',start, end )
df.head()

df = df[['open_price', 'high_price', 'low_price', 'volume']]
df['open'] = df['open_price'].shift(1)
