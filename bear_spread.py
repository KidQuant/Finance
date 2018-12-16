import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%pylab inline

sns.set_style('whitegrid')

def put_payoff(sT, strike_price, premium):
     return np.where(sT < strike_price, strike_price - sT, 0) - premium

# Initial Stock Price
 s0 = 135

 # Long Put
 strike_price_long_put = 130
 premium_long_put = 5

 # Short Put
 strike_price_short_put = 120
 premium_short_put = 2

 # Range of put option at expiry
 sT = np.arange(100,150,1)


long_put_payoff = put_payoff(sT, strike_price_long_put, premium_long_put)

fig, ax = plt.subplots()

ax.spines['bottom'].set_position('zero')
ax.plot(sT, long_put_payoff, color ='g')
ax.set_title('Long 165 Strike Put')
plt.xlabel('Stock Price (sT)')
plt.ylabel('Profit & Loss')
plt.show()

short_put_payoff = put_payoff(sT, strike_price_short_put, premium_short_put) * -1.0

fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT, short_put_payoff, color ='r')
ax.set_title('Short 155 Strike Put')
plt.xlabel('Stock Price (sT)')
plt.ylabel('Profit & Loss')
plt.show()

Bear_Put_payoff = long_put_payoff + short_put_payoff

fig, ax = plt.subplots(figsize=(9,5))
ax.spines['bottom'].set_position('zero')
ax.plot(sT, Bear_Put_payoff, color ='b', label = 'Bear Put Spread')
ax.plot(sT, long_put_payoff,'--', color ='g', label ='Long Put')
ax.plot(sT, short_put_payoff,'--', color ='r', label ='Short Put')
plt.legend()
plt.xlabel('Stock Price (sT)')
plt.ylabel('Profit & Loss')
plt.show()
