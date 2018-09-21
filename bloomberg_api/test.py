import pdblp
import blpapi
options = blpapi.SessionOptions()
options.setServerHost('localhost')
options.setServerPort(8194)
session = blpapi.Session(options)
session.start()

import pandas as pd

con = pdblp.BCon(debug = True, port=8194, timeout = 5000)

con.start()

 = con.bdh('SPY US Equity', 'PX_LAST', '20150629', '20150630')

con.bdh('SPY US Equity', ['PX_LAST', 'VOLUME'], '20170101','20180920')

con.ref_hist('AUD1M Curncy', 'DAYS_TO_MTY', dates = ['20150625', '20150626'])

import joblib
import shutil
from tempfile import mkdtemp
