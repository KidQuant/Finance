
#forecast.def

from __future__ import print_function

import datetime
import numpy as np
import pandas as pd
import sklearn
import fix_yahoo_finance as yf

from pandas_datareader import data as pdr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC

yf.pdr_override()

def create_lagged_series(symbol, start_date, end_date, lags = 5):
    """
    This creates a pandas DataFrame that stores the
    percentage returns of the adjusted closing value of
    a stock obtained from Yahoo Finance, along with a
    number of lagged returns from the prior trading days
    (lags defaults to 5 days). Trading volume, as well as
    the Direction from the previous day, are also included.
    """

    #Obtain stock information from Yahoo Finance

    ts = pdr.get_data_yahoo(
            symbol,
            start_date-datetime.timedelta(days=365),
            end_date
    )

    #Create the new lagged DataFrame

    tslag = pd.DataFrame(index = ts.index)
    tslag['Today'] = ts['Adj Close']
    tslag['Volume'] = ts['Volume']

    # Create the shifted lag series of prior trading period close values
    for i in range(0, lags):
        tslag['lag%s' % str(i+1)] = ts['Adj Close'].shift(i+1)

    # Create the return DataFrame
    tsret = pd.DataFrame(index = tslag.index)
    tsret['Volume'] = tslag['Volume']
    tsret['Today'] = tslag['Today'].pct_change() * 100.0

    #If any of the values of percentage return equal zero, set them to
    # a small number (stops issues with QDA model in scikit-learn)
    for i, x enumerate(tsret['Today']):
        if (abs(x) < 0.0001):
            tsret['Today'][i] = 0.0001

    # Create the lagged percentage returns column
    for i in range(0, lags):
        tsret['Lag%s' % str(i+1)] = \
        tslag['Lag%s' % str(i+1)].pct_change()*100.0

    # Create the "Direction" column (+1 of -1) indicating an up/down day
    tsret['Direction'] = np.sign(tsret['Today'])
    tsret = tsret[tsret.index >= start_date]

    return tsret

if __name__ == '__main__':
    #Create a lagged series of SUP 500 US stock market index
    snpret = create_lagged_series(
    '^GSPC', datetime.datetime(2010,1,10),
    datetime.datetime(2018,09,31), lags = 5
    )

    #Use the prior two days of returns as predictor
    # values, with direction as the response
    X = snpret[['Lag1', 'Lag2']]
    y = snpret[['Direction']]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2010,1,10)

    #Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    # Create the (parametrised) models
    print('Hit Rates/Confusion_Matrices:\n')
    models = [('LR', LogisticRegression()),
              ('LDA', LinearDiscriminantAnalysis()),
              ('QDA', QuadraticDiscriminantAnalysis()),
              ('LSVC', LinearSVC()),
              ('RSVM', SVC(
                C = 1000000.0, cache_size = 200, class_weight = None,
                coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
                max_iter=-1, probability=False, random_state=None,
                shrinking=True, tol=0.001, verbose = False)
              ),
              ('RF', RandomForestClassifier(
                n_estimators=1000, criterion='gini',
                max_depth=None, min_samples_split=2,
                min_samples_leaf =1, max_features = 'auto',
                bootstrap = True, oob_score = False, n_job = 1,
                random_state = None, verbose = 0)
              )]

for m in models:
