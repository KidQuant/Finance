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

df = df[['open_price', 'high_price', 'low_price', 'close_price','volume']]
df['open'] = df['open_price'].shift(1)
df['high'] = df['high_price'].shift(1)
df['low'] = df['low_price'].shift(1)
df['close'] = df['close_price'].shift(1)
df['volume'] = df['volume'].shift(1)

X = df[['open', 'high', 'low', 'close']]
y = df['close_price']

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

steps = [('imputation', imp),
         ('scaler', StandardScaler()),
         ('lasso', Lasso())]

pipeline = Pipeline(steps)

parameters = {'lasso_alpha':np.arange(0.0001, 10,.0001),
              'lasso_max_iter':np.random.uniform(100,100000,4)}

reg = rcv(pipeline, parameters, cv=5)

for t in np.arange(50,97,3):
    get_ipython().magic('reset_selective -f reg1')
    split = int(t*len(X)/100)
    reg.fit(X[:split],y[:split])
    best_alpha = reg.best_params_['lasso_alpha']
    best_iter = reg.best_paras_['lasso_max_iter']
    reg1 = Lasso(alpha=best_alpha, max_iter=best_iter)
    X = imp.fit_transform(X,y)
    reg1.fit(X[:split],y[:split])

    df['P_C_%i'%t] =0.
    df.iloc[:df.columns.get_loc('P_C_%'%t)]=regl.predict(x[:])
    df['Error_%i'%t] = np.abs(df['P_C_%i'%t] - df['close'])

    e = np.mean(df['Error_%i'&t][split:])
    train_e = np.mean(df['Error_%i'%t][:split])
    avg_err[t] = e
    avg_train_err[t] = train_e
Range = df['high'][split:] - df['low'][split:]
