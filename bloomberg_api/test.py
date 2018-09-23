import pdblp
import blpapi
options = blpapi.SessionOptions()
options.setServerHost('localhost')
options.setServerPort(8194)
session = blpapi.Session(options)
session.start()
import pandas as pd
import tia.bbg.datamgr as dm



tickers = ['IBM US Equity', 'AAPL US Equity']

fields = ['PX_Open', 'PX_High', 'PX_Low']

con = pdblp.BCon(debug = False, port=8194, timeout = 5000)

con.start()

con.bdh('SPY US Equity', 'PX_LAST', '20150629', '20150630')

con.bdh('SPY US Equity', ['PX_LAST', 'VOLUME'], '20170101','20180920')

con.bdh('MPMIEZMA Index',  ' PX_LAST', '20150101', '20180917')

con.ref_hist('AUD1M Curncy', 'DAYS_TO_MTY', dates = ['20150625', '20150626'])

con.ref(['AUDUSD Curncy', 'NZDUSD Curncy'], 'SETTLE_DT')

con.ref(['AUDUSD Curncy', 'NZDUSD Curncy'], ['SETTLE_DT', 'DAYS_TO_MTY'])

import joblib
import shutil
from tempfile import mkdtemp

temp_dir = mkdtemp()
cacher = joblib.Memory(temp_dir)
bdh = cacher.cache(con.bdh, ignore=['self'])
spy = bdh('SPY US Equity', 'PX_LAST', '20150101','20180901' )

df = con.bdh(['SPY Equity', 'IWM Equity'], 'PX_LAST', '20150101', '20180303')

df.head()
df.drop('date', axis=1)

print(con.bdh.__doc__)

df = con.bdh(['IBM US Equity', 'MSFT US Equity'], ['PX_LAST', 'OPEN'],'20061227', '20061231', elms=[("periodicityAdjustment", "ACTUAL")])

df['date']
