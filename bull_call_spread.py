import numpy as np
import matplotlib.pyplot as plt
%pylab inline
import seaborn as sns
import math

sns.set_style('whitegrid')

#Bull Call Spread

def call_payoff(sT, strike_price, premium):
    return np.where(sT > strike_price, sT - strike_price, 0) - premium

spot_price = 150

# Long call
strike_price_long_call = 145
premium_long_call = 8.88

# Short call
strike_price_short_call = 155
premium_short_call = 3.99

# Stock price range at expiration of the call
sT = np.arange(0.95*spot_price,1.1*spot_price,1)

payoff_long_call = call_payoff(sT, strike_price_long_call, premium_long_call)
# Plot
fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT,payoff_long_call,label='Long 920 Strike Call',color='g')
plt.xlabel('Infosys Stock Price')
plt.ylabel('Profit and loss')
plt.legend()
plt.show()

payoff_short_call = call_payoff(sT, strike_price_short_call, premium_short_call) * -1.0
# Plot
fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT,payoff_short_call,label='Short ' + str(strike_price_short_call) + ' Strike Call',color='r')
plt.xlabel('Infosys Stock Price')
plt.ylabel('Profit and loss')
plt.legend()
plt.show()

payoff_bull_call_spread = payoff_long_call + payoff_short_call

print("Max Profit:", max(payoff_bull_call_spread))
print("Max Loss:", min(payoff_bull_call_spread))
# Plot
fig, ax = plt.subplots(figsize = (9,5))
ax.spines['bottom'].set_position('zero')
ax.plot(sT,payoff_long_call,'--',label='Long ' + str(strike_price_long_call) +' Strike Call',color='g')
ax.plot(sT,payoff_short_call,'--',label='Short ' + str(strike_price_short_call) + ' Strike Call ',color='r')
ax.plot(sT,payoff_bull_call_spread,label='Bull Call Spread')
plt.xlabel('Infosys Stock Price')
plt.ylabel('Profit and loss')
plt.legend()
plt.savefig('bull_call_spread.png', bbox_inches = 'tight')
