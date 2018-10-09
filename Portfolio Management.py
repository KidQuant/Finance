import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objs as go
%matplotlib inline
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__)

yf.pdr_override()

start_sp = datetime.datetime(2013,1,1)
end_sp = datetime.datetime(2018,3,9)

end_of_last_year = datetime.datetime(2017,12,29)

stocks_start = datetime.datetime(2013,1,1)
stocks_end = datetime.datetime(2018,3,9)

sp500 = pdr.get_data_yahoo('^GSPC', start_sp, end_sp)
