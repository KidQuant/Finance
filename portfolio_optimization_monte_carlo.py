import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
import datetime as dt

yf.pdr_override()

stock = ['BAC', 'GS', 'JPM', 'MS']

start = dt.datetime(2017,1,1)
end = dt.datetime.now()

data = pdr.get_data_yahoo(stock, start, end )['Adj Close']

data =data.iloc[::-1]
print(data.round(2))

stock_ret = data.pct_change()
print(stock_ret.round(4)*100)

mean_returns = stock_ret.mean()
cov_matrix = stock_ret.cov()
print(mean_returns)
print(cov_matrix)

num_iterations = 10000
simulation_res = np.zeros((4+len(stock)-1,num_iterations))

for i in range(num_iterations):
    weights = np.array(np.random.random(4))
    weights /= np.sum(weights)
    portfolio_return = np.sum(mean_returns *weights)
    portfilio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    simulation_res[0,i] = portfolio_return
    simulation_res[1,i] = portfilio_std_dev
    simulation_res[2,i] = simulation_res[0,i] / simulation_res[1,i]
    for j in range(len(weights)):
        simulation_res[j+3,i] = weights[j]

sim_frame = pd.DataFrame(simulation_res.T, columns=['ret','stdev','sharpe',stock[0],stock[1],stock[2],stock[3]])
print(sim_frame.head(5))
print(sim_frame.tail(5))

#Spot the position of the portfolio with the highest Sharpe Ratio
max_sharpe = sim_frame.iloc[sim_frame['sharpe'].idxmax()]

#Spot the position of the portfolio with the minimum Standard Deviation
min_std = sim_frame.iloc[sim_frame['stdev'].idxmin()]
print('The portfolio for max Sharpe Ratio:\n', max_sharpe)
print('The portfolio for min risk:\n', min_std)

plt.scatter(sim_frame.stdev,sim_frame.ret,c=sim_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')
plt.ylim(0,.003)
plt.xlim(0.0075,0.012)
plt.scatter(max_sharpe[1],max_sharpe[0],marker=(5,1,0),color='r',s=600)
plt.scatter(min_std[1],min_std[0],marker=(5,1,0),color='b',s=600)
