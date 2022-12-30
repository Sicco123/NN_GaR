import numpy as np

def prepare_forecast_target(horizon_list, y):
    max_horizon = np.max(horizon_list)
    y_star = y[max_horizon:]
    return y_star

def prepare_forecast_predictors(horizon_list, h, X):
    max_horizon = np.max(horizon_list)
    X_star = X[max_horizon-h:-h]
    return X_star
