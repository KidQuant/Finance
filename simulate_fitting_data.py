
#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
<<<<<<< HEAD
import seaborn as sns
import pymc3 as pm

sns.set(style='darkgrid', palette='muted')
=======
import pymc3 as pm
import seaborn as sns


sns.set(style="darkgrid", palette="muted")

#%%
>>>>>>> 05a79d44c8b519276b4743330866a1ef3aacf6bc

def simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq):
    """
    Simulate a random dataset using a noisy
    linear process.

    N: Number of data points to simulate
    beta_0: Intercept
    beta_1: Slope of univariate predictor, X
    """
<<<<<<< HEAD

    # Create a pandas DataFrame with column ’x’ containing
    # N uniformly sampled values between 0.0 and 1.0

    df = pd.DataFrame(
        {"x":
        np.random.RandomState(42).choice(
            map(
                lambda x: float(x)/10000,
                np.arange(10000)
=======
    # Create a pandas DataFrame with column 'x' containing
    # N uniformly sampled values between 0.0 and 1.0
    df = pd.DataFrame(
        {"X": 
            np.random.RandomState(42).choice(
                map(
                    lambda x: float(x)/100.0, 
                    np.arange(N)
>>>>>>> 05a79d44c8b519276b4743330866a1ef3aacf6bc
                ), N, replace=False
            )
        }
    )

<<<<<<< HEAD
    # Use a linear model (y ~ beta_0 + beta_1*x + epsilon) to
    # generate a column ’y’ of responses based on ’x’
    eps_mean = 0.0
    df["y"] = beta_0 + beta_1*df["x"] + np.random.RandomState(42).normal(
=======
    # Use a linear model (y ~ beta_0 + beta_1*x + epsilon) to 
    # generate a column 'y' of responses based on 'x'
    eps_mean = 0.0
    df["Y"] = beta_0 + beta_1*df["X"] + np.random.RandomState(42).normal(
>>>>>>> 05a79d44c8b519276b4743330866a1ef3aacf6bc
        eps_mean, eps_sigma_sq, N
    )

    return df

<<<<<<< HEAD

=======
>>>>>>> 05a79d44c8b519276b4743330866a1ef3aacf6bc
#%%

def glm_mcmc_inference(df, iterations=5000):
    """
    Calculates the Markov Chain Monte Carlo trace of
<<<<<<< HEAD
    a Generalized Linear Model Bayesian linear regression
=======
    a Generalised Linear Model Bayesian linear regression 
>>>>>>> 05a79d44c8b519276b4743330866a1ef3aacf6bc
    model on supplied data.

    df: DataFrame containing the data
    iterations: Number of iterations to carry out MCMC for
    """
<<<<<<< HEAD
    basic_model = pm.Model()
    with basic_model:
        # Create the glm using Patsy model syntax
        # We use a Normal distribution for the likelihood

        pm.glm.glm('y ~ x', df, family=pm.glm.families.Normal())

        # Use Maximum A Posterior (MAP) optimization
        # as initial value for MCMC

=======
    # Use PyMC3 to construct a model context
    basic_model = pm.Model()
    with basic_model:
        # Create the glm using the Patsy model syntax
        # We use a Normal distribution for the likelihood
        pm.glm.glm("y ~ x", df, family=pm.glm.families.Normal())

        # Use Maximum A Posteriori (MAP) optimisation as initial value for MCMC
>>>>>>> 05a79d44c8b519276b4743330866a1ef3aacf6bc
        start = pm.find_MAP()

        # Use the No-U-Turn Sampler
        step = pm.NUTS()

<<<<<<< HEAD
        #Calculate the trace
        trace = pm.sample(
            iterations, step, start,
=======
        # Calculate the trace
        trace = pm.sample(
            iterations, step, start, 
>>>>>>> 05a79d44c8b519276b4743330866a1ef3aacf6bc
            random_seed=42, progressbar=True
        )

    return trace

<<<<<<< HEAD


=======
>>>>>>> 05a79d44c8b519276b4743330866a1ef3aacf6bc
#%%

if __name__ == "__main__":
    # These are our "true" parameters
<<<<<<< HEAD
    beta_0 = 1.0 # Intercept
    beta_1 = 2.0 # Slope

    # Simulate 100 data points, with a variance of 0.5
    N = 200
=======
    beta_0 = 1.0  # Intercept
    beta_1 = 2.0  # Slope

    # Simulate 100 data points, with a variance of 0.5
    N = 100
>>>>>>> 05a79d44c8b519276b4743330866a1ef3aacf6bc
    eps_sigma_sq = 0.5

    # Simulate the "linear" data using the above parameters
    df = simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq)

    # Plot the data, and a frequentist linear regression fit
    # using the seaborn package
    sns.lmplot(x="x", y="y", data=df, size=10)
    plt.xlim(0.0, 1.0)
<<<<<<< HEAD

    trace = glm_mcmc_inference(df, iterations = 5000)
    pm.traceplot(trace[500:])
    plt.show()

    #plot a sample of posterior regression lines
    sns.lmplot(x='x', y='y', data=df, size=10, fit_reg=False)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 4.0)
    pm.glm.plot_posterior_predictive(trace,samples=100)
    x = np.linspace(0,1,N)
    y = beta_0 + beta_1*x
    plt.plot(x,y, label='True Regression Line', lw=3.,c='green')
    plt.legend(loc=0)
    plt.show()
=======
    
    trace = glm_mcmc_inference(df, iterations=5000)
    pm.traceplot(trace[500:])
    plt.show()

    # Plot a sample of posterior regression lines
    sns.lmplot(x="x", y="y", data=df, size=10, fit_reg=False)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 4.0)
    pm.glm.plot_posterior_predictive(trace, samples=100)
    x = np.linspace(0, 1, N)
    y = beta_0 + beta_1*x
    plt.plot(x, y, label="True Regression Line", lw=3., c="green")
    plt.legend(loc=0)
    plt.show()

#%%

if __name__ == "__main__":
    # These are our "true" parameters
    beta_0 = 1.0  # Intercept
    beta_1 = 2.0  # Slope

    # Simulate 100 data points, with a variance of 0.5
    N = 100
    eps_sigma_sq = 0.5

    # Simulate the "linear" data using the above parameters
    df = simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq)

    # Plot the data, and a frequentist linear regression fit
    # using the seaborn package
    sns.lmplot(x="x", y="y", data=df, size=10)
    plt.xlim(0.0, 1.0)
>>>>>>> 05a79d44c8b519276b4743330866a1ef3aacf6bc
