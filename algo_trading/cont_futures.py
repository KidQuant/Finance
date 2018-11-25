#cont_futures.py

from __future__ import print_function

import datetime

import numpy as np
import pandas as pd
import quandl

<<<<<<< HEAD
def futures_rollover_weights(start_date, expiry_dates, contracts, rollover_days = 5):
    """This constructs a pandas DataFrame that contains weights (between 0.0 and 1.0)
    of contract positions to hold in order to carry out a rollover of rollover_days
    prior to the expiration of the earliest contract. The matrix can then be
    'multiplied' with another DataFrame containing the settle prices of each
    contract in order to produce a continuous time series futures contract."""

    #constructs a sequence of dates beginning from the earliest contract start
    # date to the end of the final contract
    dates = pd.date_range(start_date, expiry_date[-1]m freq='B')

    #Create the 'roll weights' DataFrame that will store the mutlipliers for
    # each contract depending upon the settlement date and rollover_days
    for i, (item, ex_date) in enumerate(expiry_dates.iteritems()):
        if i < len(expiry_dates) -1:
            roll_weights.ix[prev_date:ex_date - pd.offset.BDay(), items] = 1
            roll_rng = pd.date_range(end=ex_date - pd.offsets.BDay(),
                                     periods = rollover_days + 1, freq ='B')

            #Create a sequence of roll weights (i.e. [0.0, 0.2,....,0.8,1.0]
            # and use these to adjust the weightings of each future
            decay_weights = np.linspace(0, 1, rollover_days + 1)
=======
def futures_rollover_weights(start_date, expiry_dates, contracts, rollover_days =5):
    """This constructs pandas DataFrame that contains weights (between 0.0 and 1.0)
    of contract position to hold in order to carry out a rollover of rollover_days
    prior to the expiration of the earliest contract. The matrix can then be
    'multipled' with another DataFrame containing the settle prices of each
    contract in order to produce a continous time series future contract."""

    #Construct a sequence of dates beginning from the earliest contract start
    # date to the end of the final contract
    dates = pd.date_range(start_date, expiry_dates[-1], freq='B')

    # Create the 'roll weights' DataFrame that will store the multipliers for
    # each contract (between 0.0 and 1.0)
    roll_weight = pd.DataFrame(np.zeros((len(dates), len(contracts))),
                               index=dates, columns=contracts)
    prev_date = roll_weights.index[0]

    # Loop through each contract and create the specific weightings for
    # each contract depending upon the settlement date and rollover_days
    for i, (item, ex_date) in enumerate(expiry_dates.iteritems()):
        if i < len(expiry_dates) - 1:
            roll_weights.ix[prev_date:ex_date - pd.offset.BDay(), item] = 1
            roll_rng = pd.date_range(end = ex_date - pd.offsets.BDay(),
                                     preiods = rollover_days + 1, freq = 'B')

            # Create a sequence of roll weights (i.e. [0.0, 0.2,...,0.8,1.0])
            # and use these to adjust the weightings of each future
            decay_weights = np.linspace(0,1, rollover_days + 1)
>>>>>>> ee260386286faea8761d64096864c2c2b5136515
            roll_weights.ix[roll_rng, item] = 1 - decay_weights
            roll_weights.ix[roll_rng, expiry_dates.index[i+1]] = decay_weights
        else:
            roll_weights.ix[prev_date:, item] = 1
        prev_date = ex_date
    return roll_weights
<<<<<<< HEAD
if __name__ == '__main__':
    #  Download the current Front and Back (near and far) futures contracts
    # for WTI Crude, traded on NYMEX, from Quandl.com. You will need to
    # adjust the contracts to reflect your current near/far
    wti_near = Quandl.get()
=======

if __name__ == '__main__':
    # Download the current Front and Back (near and far) futures contracts
    # for WTI Crude, traded on NYMEX, from Quandl.com. You will need to
    # adjust the contract to reflect your current near/far contract
    # depending upon the point at which you read this!
    wti_near = quandl.get('OFDP/FUTURE_CLF2016')
    wti_far = quandl.get('OFDP/FUTURE_CLG2016')
    wti = pd.DataFrame({'CLF2014': wti_near['Settle'],
                        'CLG2014': wti_far['Settle']}, index = wti_far.index)

    # Create the dictionary of expiry date for each each contract
    expiry_dates = pd.Series({'CLF2016': datetime.datetime(2016,1,1),
                              'CLG2016': datetime.datetime(2016,3,1)}).order()

    # Obtain the rollover weighting matrix/DataFrame
    weights = futures_rollover_weights(wti_near.index[0], expiry_dates, wti.columns)

    # Construct the continous future of the WTI CL contracts
    wti_cts = (wti * weights).sum(1).dropna()

    # Output the merged series of contract settle prices
    print(wti_cts.tail(60))
>>>>>>> ee260386286faea8761d64096864c2c2b5136515
