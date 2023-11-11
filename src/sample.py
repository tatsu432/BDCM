import numpy as np
import conf

def sample_array_u(d):
    array_u = np.array([])
    for i in range(d):
        u = np.random.normal(conf.mu, conf.sigma, conf.n_obs)
        if i == 0:
            array_u = u
        else:
            array_u = np.vstack((array_u, u))
    return array_u