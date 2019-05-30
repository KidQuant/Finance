from keras.layers import Dense, LSTM, Activation, Dropout, Flatten
from keras.models import Sequential

from sklearn import svm, metrics, preprocessing
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Show matplotlib plots inline (nicely formatted in the notebook)
import matplotlib.pyplot as plt

def xrange(x):
    return iter(range(x))

class StockPredictor:

    def __init__(self):
        self.tickerSymbol = ''

    #LoadData call queries the API, caching if necessary
    def loadData(self, tickerSymbol, startDate, endDate, reloadData=False, fileName='stockData.csv'):

        self.tickerSymbol = tickerSymbol
        self.startDate = startDate
        self.endDate = endDate

        if reloadData:
            #get data from yahoo finance to tickerSymbol
            data = pdr.get_data_yahoo(ticker, start, end)

            # Save a csv to stop blowing up their API
            data.to_csv(fileName)

            #save then reload as the yahoo finance date doesn't load right in Pandas
            data = pd.read_csv(fileName)
        else:
            data = pd.read_csv(fileName)

        # Due to differing markets and timezones, the public holidays etc (e.g. BHP being an
        # Australian stock, ASX doesn't open on 26th Jan due to National Holiday) there are some
        # gaps in the data. From manual inspection and knowledge of the dataset, its safe to take the
        # previous days' value
        data.fillna(method='ffill', inplace=True)
        self.data = data

    def setData(self, data, tickerSymbol, startDate, endDate):
        self.tickerSymbol = tickerSymbol
        self.startDate = startDate
        self.endDate = endDate
        self.data = data

        # prepare data call does the preprocessing of the loaded data
        # sequence length is a tuning paramter - this is the length of the sequence that will
        # be trained on. Too long and too short and the algorithm won't be able to find any trend
        # set as 5 days by default, and this works pretty well
        def prepareData(self, predictDate, metric = 'Adj Close', sequenceLength = 5):

            # number of day to predict ahead
            predictDate = dt.datetime.strptime(predictDate, '%Y-%m-%d')
            endDate = dt.datetime.strptime(self.endDate, '%Y-%m-%d')

            #this pandas gets the number of business days ahead
