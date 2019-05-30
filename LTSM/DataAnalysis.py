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

start = dt.datetime(2015,1,1)
end = dt.datetime.now()

pdr.get_data_yahoo('TSLA', start, end)

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
    p1 = abs(predictedPrice - actualPrice)
    return (p1/t1) * 100.0

from keras.layers import Dense, LSTM, Activation, Dropout, Flatten
from keras.models import Sequential
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

startDate = '2004-08-19'
endDate = '2019-05-30'
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
        predictDate = predictDate
        endDate = self.endDate

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
predicted = sp.predictSVR()
print("Actual:", actual, "Predicted by SVR", predicted)
print("Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0))
print("Variance of return:{:.4f} %".format(endDate,actual,predicted))

# %%

import os
import time
import warnings
from numpy import newaxis
from keras.layers.recurrent import LSTM
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings('ignore') #Hide messy Numpy warnings

def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    print(model.summary())
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.savefig('results.jpg')
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.savefig('multipleResults.jpg')
    plt.show()

epochs  = 1
seq_len = 50

print('> Loading data... ')

X_train, y_train, X_test, y_test = load_data('stock_dfs/Google.csv', seq_len, True)

print('> Data Loaded. Compiling...')

model = build_model([1, 50, 100, 1])

model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=epochs,
    validation_split=0.05)

predicted = predict_point_by_point(model, X_test)
plot_results(predicted, y_test)

predictions = predict_sequences_multiple(model, X_test, seq_len, 50)
plot_results_multiple(predictions, y_test, 50)
