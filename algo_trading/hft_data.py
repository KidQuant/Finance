
#hft_data.py

from __future__ import print_function
from abc import ABCMeta, abstractmethod
import datetime
import os, os.path

import numpy as np
import pandas as pd

from event import MarketEvent
from data import DataHandler

class HistoricCSVDataHandlerHFT(DataHandler):
    """
    HistoricCSVDataHandlerHFT is desgined to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface.
    """
    def __init__(self, events, csv_dir, symbol_list):
        """
        Initializes the historic data handler by requesting
        the location of the CSV files and a list of symbols.

        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.

        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol string.
        """

        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list =  symbol_list

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.bar_index = 0

        self.open_convert_csv_files()

    def _open_convert_csv_files():
        """
        Open the CSV files from the data directory, converting
        them into pandas DataFrame within a symbol dictionary.

        For this handler it will be assumed that the data is
        taken from Yahoo. This its format will be respected.
        """
        comb_index = None
        for s in self.symbol_list:
            # Load the CSV file with no header information, indexed on date
            self.symbol_data[s] = pd.io.parsers.read_csv(
                os.path.join(self.csv_dir, '%s.csv' % s),
                header = 0, index_col = 0, parse_dates = True,
                names = [
                    'datetime', 'open', 'low',
                    'high',
                ]
            ).sort()

            # Combine the index to pad forward values
            if comb_index is None:
                comb_indx = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol[s].index)

            #Set the latest symbol_data to None
            self.latest_symbol_data[s] = []

        for s in self.symbol_list:
            self.symbol_data[s] =  self.symbol_data[s].reindex(
                index = comb_index, method = 'pad'
            )
            self.symbol_data[s]['returns'] = self.symbol_data[s]['close'].pct_change()
            self.symbol_data[s] = self.symbol_data[s].iterrows()

    def _get_new_bars(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        for b in self.symbol_data[symbol]:
            yield b

    def get_latest_bar(self, symbol):
        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            
