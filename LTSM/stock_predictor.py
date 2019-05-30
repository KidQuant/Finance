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

fileName = "LTSM/backTest.csv"
ticker = 'GOOG'
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

startDate = '2004-08-19'
endDate = '2017-03-01'
queryDate = '2019-09-07'
tickerSymbol = 'GOOG'
metric = 'Adj Close'
fileName = 'stock_dfs/GOOG.csv'

# Show matplotlib plots inline (nicely formatted in the notebook)
import matplotlib.pyplot as plt

def xrange(x):
    return iter(range(x))

class StockPredictor:

    def __init__(self):
        self.tickerSymbol = ''

    #LoadData call queries the API, caching if necessary
    def loadData(self, tickerSymbol, startDate, endDate, reloadData=False, fileName='stock_dfs/stockData.csv'):

        self.tickerSymbol = tickerSymbol
        self.startDate = startDate
        self.endDate = endDate

        if reloadData:
            #get data from yahoo finance fo tickerSymbol
            data = pdr.get_data_yahoo(tickerSymbol, startDate, endDate)

            # save as CSV to stop blowing up their API
            data.to_csv(fileName)

            # save then reload as the yahoo finance date doesn't load right in Pandas
            data = pd.read_csv(fileName)
        else:
            data = pd.read_csv(fileName)

        # Due to differing markets and timezones, public holidays etc (e.g. BHP being an Australian stock,
        # ASX doesn't open on 26th Jan due to National Holiday) there are some gaps in the data.
        # from manual inspection and knowledge of the dataset, its safe to take the previous days' value
        data.fillna(method='ffill', inplace=True)
        self.data = data

    def setData(self, data, tickerSymbol, startDate, endDate):
        self.tickerSymbol = tickerSymbol
        self.startDate = startDate
        self.endDate = endDate
        self.data = data

    #preparedata call does the preprocessing of the loaded data
    #sequence length is a tuning parameter - this is the length of the sequence that will be trained on.
    # Too long and too short and the algorithms won't be able to find any trend - set as 5 days
    # by default, and this works pretty well
    def prepareData(self, predictDate, metric = 'Adj Close', sequenceLength=5):

        # number of days to predict ahead
        predictDate = dt.datetime.strptime(predictDate, "%Y-%m-%d")
        endDate = dt.datetime.strptime(self.endDate, "%Y-%m-%d")

        #this pandas gets the number of business days ahead, within reason ( i.e. doesn't know about local market
        #public holidays, etc)
        self.numBdaysAhead = abs(np.busday_count(predictDate, endDate))
        print ("business days ahead", self.numBdaysAhead)

        self.sequenceLength = sequenceLength
        self.predictAhead = self.numBdaysAhead
        self.metric = metric

        data = self.data
        # Calculate date delta
        data['Date'] = pd.to_datetime(data['Date'])
        data['date_delta'] = (data['Date'] - data['Date'].min()) / np.timedelta64(1, 'D')

        #create the lagged dataframe
        tslag = pd.DataFrame(index=data.index)

        #use the shift function to get the price x days ahead, and then transpose
        for i in range(0, sequenceLength + self.numBdaysAhead):
            tslag["Lag%s" % str(i + 1)] = data[metric].shift(1 - i)

        #shift (-2) then corrects the sequence indexes
        tslag.shift(-2)
        tslag['date_delta'] = data['date_delta']

        # create the dataset.  This will take the first [sequenceLength] columns as the data, and the
        # value at end of sequence + number days ahead as the label
        trainCols = ['date_delta']
        for i in range(0, sequenceLength):
            trainCols.append("Lag%s" % str(i + 1))
        labelCol = 'Lag' + str(sequenceLength + self.numBdaysAhead)

        # get the final row for predictions
        rowcalcs = tslag[trainCols]
        rowcalcs = rowcalcs.dropna()

        #need an unscaled version for the RNN
        self.final_row_unscaled = rowcalcs.tail(1)


        #due to the way the lagged set is created, there will be some rows with nulls for where
        #the staggering has not worked to to predicting too far back, or ahead.
        #  We can drop these without losing any information as these sequences will be represented in
        #other rows within the dataset
        tslag.dropna(inplace=True)

        label = tslag[labelCol]
        new_data = tslag[trainCols]

        # print ("NEW DATA", new_data.tail(1))
        #scale the data for the Linear Regression, SVR and Neural Net
        self.scaler = preprocessing.StandardScaler().fit(new_data)
        scaled_data = pd.DataFrame(self.scaler.transform(new_data))

        # print ("SCALED DATA", scaled_data.tail(1))
        self.scaled_data = scaled_data
        self.label = label

    #Linear Regression trainer
    def trainLinearRegression(self):
        lr = LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(self.scaled_data, self.label, test_size=0.25,
                                                            random_state=42)

        parameters = {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}

        grid = GridSearchCV(lr, parameters, cv=None)
        grid.fit(X_train, y_train)
        predicttrain = grid.predict(X_train)
        predicttest = grid.predict(X_test)
        print ("R2 score for training set (Linear Regressor): {:.4f}.".format(r2_score(predicttrain, y_train)))
        print ("R2 score for test set (Linear Regressor): {:.4f}.".format(r2_score(predicttest, y_test)))
        self.model = grid

    #predict Linear Regression
    def predictLinearRegression(self):
        inputSeq = self.scaler.transform(self.final_row_unscaled)
        inputSeq = pd.DataFrame(inputSeq)
        predicted = self.model.predict(inputSeq)[0]
        return predicted


    def trainSVR(self):
        clf = svm.SVR()
        X_train, X_test, y_train, y_test = train_test_split(self.scaled_data, self.label, test_size=0.25, random_state=42)

        parameters = {'C': [1, 10], 'epsilon': [0.1, 1e-2, 1e-3]}
        r2_scorer = metrics.make_scorer(metrics.r2_score)

        grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=r2_scorer)
        grid_obj.fit(X_train, y_train)
        print ("best svr params", grid_obj.best_params_)

        predicttrain = grid_obj.predict(X_train)
        predicttest = grid_obj.predict(X_test)

        print ("R2 score for training set (SVR): {:.4f}.".format(r2_score(predicttrain, y_train)))
        print ("R2 score for test set (SVR): {:.4f}.".format(r2_score(predicttest, y_test)))
        self.model = grid_obj

    def predictSVR(self):
        inputSeq = self.scaler.transform(self.final_row_unscaled)
        inputSeq = pd.DataFrame(inputSeq)
        predicted = self.model.predict(inputSeq)[0]
        return predicted

#create new stock predictor.  Argument as placeholder for future expansion
sp = StockPredictor()

#loadData will query the Qandl API with the parameters
sp.loadData(tickerSymbol, startDate, endDate, reloadData=False, fileName=fileName)

#sp.setData(data, ticker, startDate, endDate)
#preparedata does the preprocessing
sp.prepareData(queryDate, metric=metric, sequenceLength=5)


#Linear Regression first
print("******* Linear Regression *******")
sp.trainLinearRegression()
predicted = sp.predictLinearRegression()
print("Actual:", actual, "Predicted by Linear Regression", predicted)
print("Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0))
print("Variance of return:{:.4f} %".format(varianceOfReturn(endDatePrice,actual,predicted)))

#SVR is used next
print("******* SVR *******")
sp.trainSVR()
predicted = sp.predictsSVR()
print("Actual:", actual, "Predicted by SVR", predicted)
print("Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0))
print("Variance of return:{:.4f} %".format(endDate,actual,predicted))
