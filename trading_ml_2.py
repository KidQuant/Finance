import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as web
import fix_yahoo_finance

matplotlib.use('TkAgg')
import numpy as np
style.use('ggplot')

start = dt.datetime(2013,1,1)
end = dt.datetime.now()

df = web.get_data_yahoo('TSLA', start, end)

print(df.head())

df.to_csv('TSLA.csv')
df = pd.read_csv('TSLA.csv', parse_dates = True, index_col = 0)
df.plot()

df['Adj Close'].plot()
plt.show()

df['100ma'] = df['Adj Close'].rolling(window=100).mean()
print(df.head())

df['100ma'] = df['Adj Close'].rolling(window = 100, min_periods =0).mean()
print(df.head())

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=5, colspan=1, sharex=ax1)
ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

import bs4 as bs
import pickle
import requests

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('.','-')
        tickers.append(ticker)

    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)

    return tickers

save_sp500_tickers()

import os

def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2013, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index('Date', inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

get_data_from_yahoo()

def compile_data():
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close':ticker}, inplace=True)
        df.drop(['High', 'Low', 'Open', 'Close','Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

        print(main_df.head())
        main_df.to_csv('sp500_joined_closes.csv')

compile_data()

def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    df_corr = df.corr()
    print(df_corr.head())
    df_corr.to_csv('sp500corr.csv')
    data1 = df_corr.values
    fig1 = plt.figure(figsize = (75,75))
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', bbox_inches='tight')
    plt.show()

visualize_data()

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

process_data_for_labels('AAPL')

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirements = 0.02
    for col in cols:
        if col > requirements:
            return 1
        if col < -requirements:
            return -1
    return 0

from collections import Counter

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    return X, y, df

extract_featuresets('AAPL')

from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)

    print('Accuracy:',confidence)
    prediction = clf.predict(X_test)
    print('Predicted class counts:', Counter(prediction))
    print()

do_ml('AAPL')

import statsmodels.api as sm

def get(tickers, startdate, enddate):
    def data(ticker):
        return(web.get_data_yahoo(ticker, start, end))
    datas = map(data, tickers)
    return(pd.concat(datas, keys = tickers, names =['Ticker', 'Date']))

tickers = ['AAPL', 'MSFT', 'IBM', 'GOOGL']

all_data = get(tickers, (2010,1,1), (2018,1,1))

daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')

daily_pct_change = daily_close_px.pct_change()

daily_pct_change.hist(bins=50, sharex= True, figsize=(12, 8))
pd.scatter_matrix(daily_pct_change, diagonal = 'kde', alpha=0.1, figsize=(12,12))
plt.show()

all_adj_close = all_data[['Adj Close']]
all_returns = np.log(all_adj_close / all_adj_close.shift(1))
aapl_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker')  == 'AAPL']
aapl_returns.index = aapl_returns.index. droplevel('Ticker')


msft_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'MSFT']
msft_returns.index = msft_returns.index.droplevel('Ticker')

return_data = pd.concat([aapl_returns, msft_returns], axis = 1)[1:]
return_data.columns = ['AAPL', 'MSFT']

X = sm.add_constant(return_data)['AAPL']

model = sm.OLS(return_data['MSFT'], X).fit()

print(model.summary())

plt.plot(return_data['AAPL'], return_data['MSFT'], 'r.')
ax = plt.axis()
x = np.linspace(ax[0], ax[1] + 0.01)
plt.plot(x, model.params[0] * x, 'b')
plt.grid(True)
plt.axis('tight')
plt.xlabel('Apple Returns')
plt.ylabel('Microsoft Returns')
plt.show()

twtr = pd.read_csv('stock_dfs/TWTR.csv')
twtr

short_window = 10
long_window = 70

signals = pd.DataFrame(index=twtr.index)
signals['signal'] = 0.0
signals['short_mavg'] = twtr['Adj Close'].rolling(window=short_window, min_periods=1, center=False).mean()
signals['long_mavg'] = twtr['Adj Close'].rolling(window=long_window, min_periods=1, center=False).mean()
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:]
                                            > signals['long_mavg'][short_window:], 1.0, 0.0)
signals['positions'] = signals['signal'].diff()

print(signals)

signals

fig = plt.figure()
ax1 = fig.add_subplot(111,  ylabel='Price in $')
twtr['Adj Close'].plot(ax=ax1, color='r', lw=2.)
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
ax1.plot(signals.loc[signals.positions == 1.0].index, signals.short_mavg[signals.positions == 1.0], '^', markersize=10, color='m')
ax1.plot(signals.loc[signals.positions == -1.0].index, signals.short_mavg[signals.positions == -1.0],'v', markersize=10, color='k')
plt.show()


#Backtesting
initial_capital = float(100000.0)
positions = pd.DataFrame(index = signals.index).fillna(0.0)
positions['TWTR'] = 100 * signals['signal']
portfolio = positions.multiply(twtr['Adj Close'], axis=0)
pos_diff = positions.diff()
portfolio['holdings'] = (positions.multiply(twtr['Adj Close'],
        axis=0)).sum(axis=1)
portfolio['cash'] = initial_capital - (pos_diff.multiply(twtr['Adj Close'],
        axis=0)).sum(axis=1).cumsum()

portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()

print(portfolio.head())

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')
portfolio['total'].plot(ax=ax1, lw=2.0)
ax1.plot(portfolio.loc[signals.positions == 1.0].index,
        portfolio.total[signals.positions == 1.0],
        '^', markersize=10, color='m')
ax1.plot(portfolio.loc[signals.positions == -1.0].index,
        portfolio.total[signals.positions == -1.0],
        'v', markersize=10, color='k')
