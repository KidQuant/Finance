import seaborn as sns; sns.set(style = 'white')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cf0 = np.array([-100, 10, 100])
npv_cf0 = np.ones(50)

for index, rate in enumerate(np.linspace(0, 0.12, 50)):
    npv_cf0[index] = np.npv(rate,cf0)

cf1 = np.array([-100, 201, -100])
npv_cf1 = np.ones(50)

for index, rate in enumerate(np.linspace(-0.2, 0.2, 50)):
    npv_cf1[index] = np.npv(rate, cf1)

fix, ax = plt.subplots(1, 2, figsize=(14,5))

ax[0].plot(np.linspace(0,0.12,50), npv_cf0, c='r', label = 'pv(r)')
ax[0].axhline(0, c='g', label = 0)
ax[0].set_xlim(0,0.12)
ax[0].set_xlabel('$C_o=-100, C_1=10, C_2=100$', fontsize = 15)
ax[0].legend()

ax[1].plot(np.linspace(-0.2, 0.2, 50), npv_cf1, c='r', label = 'pv(r)')
ax[1].axhline(0, c='g', label = 0)
ax[1].set_xlim(-0.2, 0.2)
ax[1].set_xlabel('$C_o=-100, C_1=201, C_2=-100$', fontsize = 15)
ax[1].legend();

r = 12 * np.log(1 + .15 / 12)
r4 = 4 * (np.exp(.12 / 4) - 1)

print(r)
print(r4)
