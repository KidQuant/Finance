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
    
