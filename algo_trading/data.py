# data.py

from __future__ import print_function

from abc import ABCMeta, abstractmethod
import datetime
import os, os.path

import numpy as np
import pandas as pd

from event import MarketEvent

class DataHandler(object):
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).

    The goal of a (derived) DataHandler object is to output a generated
    set of bats (OHLCVI) for each symbol requested.

    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe." Thus a historic and live
    system will be treated identically by the rest of the backtesting suite.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bar(self, symbol):
        """
        Returns the last bar updated.
        """
        raise NotImplementedError('Should implement get_latest_bar()')

    @abstractmethod
    def get_latest_bar(self, symbol, N=1):
        """
        Returns the last N bars updated.
        """
        raise NotImplementedError('Should implement get_latest_bar')

    @abstractmethod
    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        """
        raise NotImplementError('Should implement get_latest_bar_datetime()')

    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns on of the Open, High, Low, Close, Volume or OI
        from the last bar.
        """
        raise NotImplementError('Should implement get_latest_bar_value()')

    @abstractmethod
    def get_latest_bar_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the latest_symbol
        list, or N-k if less available.
        """
        raise NotImplementError('Should implement get_latest_bar_values()')

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bars to the bars_queue for each symbol
        in a tuple OHLCVI format: (datetime, open, high, low,
        close, volume, open interest).
        """
        raise NotImplementedError('Should implement update_bars()')

class HistoricCSVDataHandler(DataHandler):
    """HistoricCSVDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface to obtain
    'latest' bar in a manner identical to a live trading interface.
    """

    def __init__(self, events, csv_dir, symbol_list):
        """
        Initializes the historic data handler by requesting
        the location of the CSV files and a list of symbols.

        It will be assumed that all files are of the form
        'symbol.csv,' where symbol is a string in the list.

        Paramaters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.bar_index = 0

        self.open_convert_csv_files()

    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.

        For this handler it will be assumed that the data is
        taken from Yahoo. Thus its dormat will be respected.
        """
        comb_index = None
        for s in self.symbol_list:
            #Load the CSV file with no header information, indexed on date
            self.symbol_data[s] = pd.io.parsers.read_csv(
                os.path.join(self.csv_dir, '%s.csv' % s),
                header = 0, index_col = 0, parse_dates =True,
                names  = [
                    'datetime', 'open', 'high',
                    'low', 'close', 'volume', 'adj_close'
                ]
            ).sort()

        #Reindex the dataframes
        for i in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].\
                reindex(index=comb_index, method = 'pad').iterrows()

    def get_new_bar(self, symbol):
        """
        Returns the last bar from the latest_sybol list.
        """
        for b in self.symbol_data[symbol]:
            yield b

    def get_latest_bar(self, symbol):
        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historicall data set.")
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("that symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print('That symbol is not available in the historical data set.')
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bar_list = self.latest_symbol_data[symbol]
        except KeyError:
            print('That symbol is not available in the historical data set.')
            raise
        else:
            return getattr(bars_list[-1][1], val_type)

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        for s in self.symbol_list:
            try:
                bar = next(self._get_new_bars(s))
            except StopIteration:
                self.latest_symbol_data[s].append(bar)
        self.events.put(MarketEvent())
