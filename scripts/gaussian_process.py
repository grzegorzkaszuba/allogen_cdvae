import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from modAL.models import BayesianOptimizer
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from typing import List


def get_gaussian_regressor(x_train, y_train):
    regressor = GaussianProcessRegressor(kernel=Matern(nu=5), alpha=5e-6, normalize_y=True, n_restarts_optimizer=5)
    regressor.fit(x_train, y_train)
    return regressor

def get_uncertainty(y_val: np.ndarray, x_val: np.ndarray, bayesian_optimizer: GaussianProcessRegressor) -> List[
    np.ndarray]:
    # Use the Bayesian optimizer (a trained Gaussian process regressor) to predict for x_val
    y_pred, sigma = bayesian_optimizer.predict(x_val, return_std=True)

    # Calculate the residuals
    residuals = y_val - y_pred.reshape(-1, 1)

    # Calculate the z-scores
    z_scores = residuals / sigma.reshape(-1, 1)

    # Calculate the p-values (two-sided)
    p_values = 2 * norm.sf(abs(z_scores))

    return z_scores, p_values
