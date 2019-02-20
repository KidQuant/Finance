from time import *
from sklearn import tree
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import time
start_time = time.time()
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

yf.pdr_override()

#trading algorithm
def algo(t,h,l,v):

    features = []
    labels = []

    for i in range(len(t) - acc):

        temp_t = t[acc + i - 1]
        temp_h = h[acc + i - 1]
        temp_l = l[acc + i - 1]
        temp_v = v[acc + i - 1]

        features.append([temp_t, temp_h, temp_l, temp_v])

        #1 means price went up
    if t[acc + i] > t[acc + i - 1]:
        labels.append([1])
    else:
        labels.append([0])

    clf = tree.DecisionTreeClassifier()
    clf.fit(features, labels)
    temp_list = []

    for i in range(acc):
        temp_list.append([])
        temp_list[i].append(t[-1*(acc - i)])
        temp_list[i].append(h[-1*(acc - i)])
        temp_list[i].append(l[-1*(acc - i)])
        temp_list[i].append(v[-1*(acc - i)])

    if clf.predict(temp_list)[0] == 1:
        return 1
    else:
        return 0

#fields
acc = 10
Points = []
Highs = []
Lows = []
Volumes = []
dates = []
CashRecords = []


Cash = 100
Bought = False
days = 0
decision = 0
stockSymbol = 'AAPL'

style.use('ggplot')
start = dt.datetime(2015,1,1)
end = dt.datetime(2019,1,1)

#importing data
df = pdr.get_data_yahoo(stockSymbol, start, end)
df.to_csv('data.csv')

for i in df[['Close']]:
    for j in df[i]:
        Points.append(round(j,2))

for i in df[['High']]:
    for j in df[i]:
        Highs.append(round(j,2))

for i in df[['Low']]:
    for j in df[i]:
        Lows.append(round(j,2))

for i in df[['Volume']]:
    for j in df[i]:
        Volumes.append(round(j,2))

for i in df[['Date']]:
    for j in df[i]:
        dates.append(dt.datetime.strptime(j,'%Y-%m-%d'))
