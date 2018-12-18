import numpy as np
import matplotlib.pyplot as plt
%pylab inline
import seaborn as sns
#Cover Call Strategy

sns.set_style('whitegrid')

price = np.arange(130,230,1) # the stock price at expiration date
strike = 160 # the strike price
premium = 7.5 # the option premium
# the payoff of short call position
payoff_short_call = [min(premium, -(i - strike-premium)) for i in price]
# the payoff of long stock postion
payoff_long_stock = [i-strike for i in price]
# the payoff of covered call
payoff_covered_call = np.sum([payoff_short_call, payoff_long_stock], axis=0)
plt.figure(figsize=(9,5))
plt.plot(price, payoff_short_call, label = 'short call')
plt.plot(price, payoff_long_stock, label = 'long stock')
plt.plot(price, payoff_covered_call, label = 'covered call')
plt.legend(fontsize = 20)
plt.xlabel('Stock Price at Expiry',fontsize = 15)
plt.ylabel('payoff',fontsize = 15)
plt.title('Covered Call Strategy Payoff at Expiration',fontsize = 20)
plt.grid(True)
plt.savefig('covered_call.png', bbox_inches = 'tight')
