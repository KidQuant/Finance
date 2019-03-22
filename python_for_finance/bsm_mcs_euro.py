#
# Monte Carlo valuation of European call options
# in Black-Scholes-Merton model
# bsm_mcs_euro.py
#

import math
import numpy as np

# Parameter Values
s0 = 100.0 # initial index level
K = 105 # strike price
T = 1.0 #time-to-maturity
r = 0.05 #riskless
sigma = 0.2 #volatility

I = 100000 # number of simulations

# Valuation Vlgorithm

z = np.random.standard_normal(I) # pseudo-random numbers
#index values at maturity
ST = s0 * np.exp((r-0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z) 
hT = np.maximum(ST - K, 0) # payoff at maturity
C0 = math.exp(-r * T) * np.mean(hT) # Monte Carlo estimator

# Result Output
print('Value of the European call option %5.3f.' % C0)

