#var.py

from __future__ import print_function

import datetime
import numpy as np
from pandas_datareader import data as web
import fix_yahoo_finance
from scipy.stats import norm

def var_cov_var(P, c, mu, sigma):
    """
    Variance-Covariance calculation of daily Value-at-Risk
    using confidence level c, with mean of returns mu &
    standard deviation of returns sigma, on a portfolio
    of value P.
    """
    alpha = norm.ppf(1-c, mu, sigma)
    return P - P*(alpha + 1)

if __name__ == "__main__":
    start = datetime.datetime(2010,1,1)
    end = datetime.datetime.now()

    suntrust = web.get_data_yahoo('STI', start, end)
    suntrust['rets'] = suntrust['Adj Close'].pct_change()

    P = 1e6
    c = 0.99
    mu = np.mean(suntrust['rets'])
    sigma = np.std(suntrust['rets'])

    var = var_cov_var(P,c,mu,sigma)
    print('Value-at-Risk: $%00.2f' % var)
