
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

def create_lagged_seris(symbol, start_date, end_date, lags = 5):
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
