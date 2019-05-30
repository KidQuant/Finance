
#%% Import packages


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Import packages

<<<<<<< HEAD
data = pd.DataFrame({'X': np.random.RandomState(42).choice(map(lambda x: float(x)/100.0, 
np.arange(10000)), 10000, replace = False)})
=======
data = pd.DataFrame({'X': np.random.RandomState(42).choice(map(lambda x: float(x)/100.0, np.arange(10000)), 10000, replace = False)})


>>>>>>> 05a79d44c8b519276b4743330866a1ef3aacf6bc

data['Y'] = 5 + 3*data['X'] + np.random.RandomState(42).normal(0.0, 0.5, 10000.0)

with pm.Model() as normal_model:
    # The prior for the data likelihood is a Normal Distribution
    family = pm.glm.families.Normal()

    # Creating the model requires a formula and data (and optionally a family)
    pm.GLM.from_formula("Y~X", data=data, family=family)

    # Perform Markov Chain Monte Carlo sampling letting PyMC choose the algorithm
    trace = pm.sample(5000, start=start, tune=1000, random_seed=42, progressbar=True)

pm.traceplot(trace[500:])
plt.show()

#Some convergence plots
fig, axes = plt.subplots(2,5, figsize=(14,6))
axes = axes.ravel()
for i in range(10):
    axes[i].hist(beta_trace[500*i:500*(i+1)])
plt.tight_layout()
plt.show()

z = geweke(trace, intervals=15)
print(z[0]['X'])
plt.scatter(*(z[0]['X']).T)
plt.hlines([-1,1], 0, 3000, linestyles='dotted')
plt.xlim(0, 3000)
plt.show()


#%%

data = pd.DataFrame({"X": np.random.RandomState(42).choice(map(lambda x: float(x)/100.0, np.arange(10000)), 10000, replace=False)})
data["Y"] = 5 + 3*data["X"] + np.random.RandomState(42).normal(0.0, 0.5, 10000)
<<<<<<< HEAD
=======


np.random.RandomState(42).choice(map(lambda x: float(x)/100.0, np.arange(10000)), 10000, replace = False)

results = np.array(map(lambda x: float(x)/100.0, np.arange(10000)))

np.random.RandomState(42).choice(results)

type(results)
>>>>>>> 05a79d44c8b519276b4743330866a1ef3aacf6bc
