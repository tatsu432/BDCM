import numpy as np
import conf
import random

def sample_array_u_and_x(d, structural_eq):
    # Sample exogenous nodes U_i ~ N(mu, sigma)
    array_u = np.array([])
    for i in range(d):
        u = np.random.normal(conf.mu, conf.sigma, conf.n_obs)
        if i == 0:
            array_u = u
        else:
            array_u = np.vstack((array_u, u))

    # Sample endogenous nodes X_i by the structural equations
    x = np.zeros([d, conf.n_obs])
    for i in range(d):
        x[i] = structural_eq(array_u[i], i, x)
    
    return array_u, x


def sample_t_and_eps_for_x(d):
  # Generate t ~ Unif([T])
  # Generate \epsilon ~ N(mu, sigma)
  t_for_x = np.array([])
  epsilon_for_x = np.array([])
  for i in range(d):
    t = np.array([random.randint(1, conf.T) for x in range(conf.n_obs)])
    epsilon = np.random.normal(conf.mu, conf.sigma, conf.n_obs)
    if i == 0:
      t_for_x = t
      epsilon_for_x = epsilon
    else:
      t_for_x = np.vstack((t_for_x, t))
      epsilon_for_x = np.vstack((epsilon_for_x, epsilon))
  return t_for_x, epsilon_for_x







# Define the function to sample when we intervene in nodes
def true_sample(d, structural_eq, ind_cause, ind_result, intervened_value_for_cause_node, array_u):
  x_do = np.zeros([d, conf.n_obs])
  for i in range(d):
     if i == ind_cause:
        x_do[i] = np.ones(conf.n_obs) * intervened_value_for_cause_node
     else:
        x_do[i] = structural_eq(array_u[i], i, x_do)
  return x_do[ind_result]