import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from pandas_datareader import data as pdr
import requests
import os
import pickle
import bs4 as bs
from pandas.plotting import scatter_matrix
import datetime as dt

yf.pdr_override()

#Show matplotlib plots inline (nicely formatted in the notebook)
%matplotlib inline

#Controll the default size of figures in this notebook
%pylab inline

pylab.rcParams['figure.figsize'] = (15, 9)

##### Entry Parameters ######
startDate = '2004-08-19'
endDate = '2017-03-01'
queryDate = '2017-02-07'
tickerSymbol = 'GOOG'
metric = 'Adj Close'

#Used for re-running: stops qurying the API if we alreadyhave the data
reloadData = True

#Stock Data - first step is to obtain the list of stocks, and then select a stock to run through machine learning

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('.','-').replace('\n','')
        tickers.append(ticker)

    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)

    return tickers

save_sp500_tickers()

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
            df = pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index('Date', inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

get_data_from_yahoo(reload_sp500=True)

data = pd.read_csv('stock_dfs/ALGN.csv', index_col = 'Date')

# Display a description of the dataset
display(data.describe())

data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)

data.plot(grid= True, subplots = True)
plt.legend(loc='best')

data.plot(secondary_y = ['Close', 'Volume'])

data.plot.scatter(x = 'Volume', y='Adj Close')

# %% Implementation: Selecting Samples

# Drop data index to be able to work with data better
data = data.reset_index(drop = True)

# Select three indices to sample from the dataset
indices = [509, 1200, 60]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)

print('Chosen samples of stock dataset')
display(samples)

# what of interests here is the percentage change from one day to the next
#data.drop(['Date'], axis = 1, inplace = True)

data = data.pct_change()

data.fillna(method = 'ffill', inplace = True)
data.fillna(method = 'bfill', inplace = True)

display(data.head())

# %% Feature relevancy

for col in list(data.columns.values):
    # Make a copy of the DataFrame, using the 'drop' function to drop the given feature
    new_data = data.drop(col, axis=1)

    # Split the data into training and testing sets using the given geature as a target
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(new_data, data[col], test_size = 0.25, random_state = 42)

    # Create a decision tree regressor and fit it to the training set
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)

    #Report the score of the prediction using the testing set
    score = regressor.score(X_test, y_test)
    print("{} R^2 score: {:2f}".format(col, score))

#Produce a scatter matrix for each pair of features in the data
scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')

#Locate number of outliers for each column, outlier being 1.5 IQR up or down from the upper or lower quartile

outliers = pd.DataFrame(index= data.index)
outliers = pd.DataFrame(np.where(
        (data > 1.5 * ((data.quantile(0.75) - data.quantile(0.25)) + data.quantile(0.75))) |
        (data < 1.5 * (data.quantile(0.25) - (data.quantile(0.75)) - data.quantile(0.25))), 1,0),
                    columns=data.columns)

#transpose the describe sso that columns can be added
res = data.describe().transpose()

display(res)

res['variance'] = data.var()
res['outliers'] = outliers.sum()
res['mean_x_outliers'] = (1 /res['outliers'])*res['mean']

display(res['variance'])
display(res['outliers'])
display(res['mean_x_outliers'])
display(res)



fileName = "backTest.csv"
ticker = 'ALGN'
start = dt.datetime(2013, 1, 1)
end = dt.datetime.now()

if reloadData:
    data  = pdr.get_data_yahoo(ticker, start, end)
    # save as CSV to stop blowing up their API
    data.to_csv(fileName)
    # save then reload as the qandl date doesn't load right in Pandas
    data = pd.read_csv(fileName)
else:
    data = pd.read_csv(fileName)

#fetch the actual price so that we can compare with what was predicted
actual = data[metric][data['Date'] == queryDate].values[0]
print("Actual price at date of query", actual)
#the endDatePrice is the price at the end of the data - used for comparison
endDatePrice = data[metric][data['Date'] == endDate].values[0]

def varianceOfReturn(endPrice, actualPrice, predictedPrice):
    t1 = abs(actualPrice - endPrice)
    p1 = abs(predictedprice - actualPrice)
    return (p1/t1) * 100.0

from keras.layers import Dense, LSTM, Activation, Dropout, Flatten
