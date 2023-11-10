import numpy as np

# # Fast
# num_seeds = 3
# n_obs = 50   # deifne the number of samples in observational data X
# n_sample_DCM = 20   # Define the number of samples we want to approximate the target

# Slow 
num_seeds = 10  # same as DCM paper
n_obs = 1000     # deifne the number of samples in observational data X
n_sample_DCM = 500    # Define the number of samples we want to approximate the target



# Slowest (same as DCM paper)
# num_seeds = 10  # same as DCM paper
# n_obs = 5000    # deifne the number of samples in observational data X
# n_sample_DCM = 1000   # same as DCM paper  # Define the number of samples we want to approximate the target




# Define T, mu, sigma
T = 100
mu = 0
sigma = 1

# Define beta_t
beta_t = (0.1 - 0.0001) * (np.linspace(1, T, T) - 1) / (T - 1) + 0.0001

# Define alpha_t
alpha_t = np.zeros(T)
for i in range(len(beta_t)):
    if i == 0:
        alpha_t[i] = (1 - beta_t[i])
    else:
        alpha_t[i] = (1 - beta_t[i]) * alpha_t[i - 1]

        
