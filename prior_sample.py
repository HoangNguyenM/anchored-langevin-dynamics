import numpy as np

def sample_prior(sample_size, d=1, method="normal", std=1):
# get the initial sample points from a chosen prior distribution
    if method == "normal":
        return np.random.normal(0,std,(sample_size,d))
    elif method == "uniform":
        return np.random.uniform(-std,std,(sample_size,d))
    elif method == "laplace":
        return np.random.laplace(0,std,(sample_size,d))
    else:
        raise NotImplementedError(f"distribution {method} unknown.")