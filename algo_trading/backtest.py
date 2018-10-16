from __future__ import print_function

import datetime
import pprint
try:
    import Queue as queue
except:
    import queue
import time

class Backtest(object):
    """
    Enscapulates the setting and components for carrying out
    an event-driven backtest.
    """

    def __init__(
        self, csv_dir, symbol_list, inital_capital,
        heartbeat, start_date, data_handler,
        execution_handler, portfolio, strategy
    ):
        """
        Initalizes the backtest.

        Parameters:
        csv_dir - The hard root to the CSV data directoryself.
        symbol_list - The list of symbol strings.
        initial_capital - The starting capital for the portfolio.
        heartbeat - Backtest "heartbeat" in seconds.
        start_date - The start datetime of the strategy.
        date_handler - (Class) Handle the market data feed.
        execution_handler - (Class) Handles the orders/fills for trades.
        portfolio - (Class) Keeps track of portfolio current and prior positions.
        strategy - (Class) Generates signals based on market data.
        """

        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.inital_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date

        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy

        self.events = quene.Queue()

        self.signels = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1

        self._generate_trading_instances()
