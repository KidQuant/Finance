
#quandl_data.py

from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd
import requests

def construct_futures_symbols(
        symbol, start_year=2010, end_year=2014
    ):
    """
    Constructs a list of futures contract codes
    for a particular symbol and timeframe.
    """
    futures = []
    ## March, June, September and
    #December delievery codes
    months = 'HMUZ'
    for y in range(start_year, end_year+1):
        for m in months:
            futures.append('%s%s%s' % (symbol, m, y))
    return futures

def download_contract_from_quandl(contract, dl_dir):
    """
    Download an individual futures contract from Quandl and then
    store it to disk in the 'dl_dir' directory. An auth_token is
    required, which is obtained from the Quandl upon sign-upself.
    """
    #Construct the API call from the contract and auth_token
    api_call = 'http://www.quandl.com/api/v1/datasets/'
    api_call += 'OFDP/FUTURE_%s.csv' % contract
    #If you wish to add an auth token for more downloads, simply
    #comment the following line and replace MY_AUTH_TOKEN with
    
