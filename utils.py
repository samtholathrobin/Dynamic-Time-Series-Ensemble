import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from numba import jit
from hmmlearn import hmm
from statsmodels.tsa.arima.model import ARIMA

def get_mae_errors(df_test, out) -> np.float64:    
    prev_days = 1
    if len(out) < prev_days:
        return (mean_absolute_error(df_test, out))
    else:
        return (mean_absolute_error(df_test[-prev_days:], out[-prev_days:]))
    
def get_mse_errors(df_test, out) -> np.float64:    
    prev_days = 1
    if len(out) < prev_days:
        return (mean_squared_error(df_test, out)) 
    else:
        return (mean_squared_error(df_test[-prev_days:], out[-prev_days:]))
    

def get_rmse_errors(df_test, out) -> np.float64:    
    prev_days = 1
    if len(out) < prev_days:
        return np.sqrt((mean_squared_error(df_test, out)))
    else:
        return np.sqrt((mean_squared_error(df_test[-prev_days:], out[-prev_days:])))
    
def get_mape_errors(df_test, out) -> np.float64:    
    prev_days = 1
    if len(out) < prev_days:
        return (mean_absolute_percentage_error(df_test, out)) 
    else:
        return (mean_absolute_percentage_error(df_test[-prev_days:], out[-prev_days:]))
    
@jit(nopython = True)
def normalized_inverse_of_errors_weighting(model1, model2, model3):
    errors = np.array([model1, model2, model3])
    weights = (1 / errors) / np.sum(1 / errors)
    return weights

@jit(nopython = True)
def softmax_weighting(model1, model2, model3, gamma=1):
    errors = np.array([model1, model2, model3])
    weights = np.exp(-gamma * errors) / np.sum(np.exp(-gamma * errors))
    return weights

@jit(nopython = True)
def proportional_weighting(model1, model2, model3, k):
    errors = np.array([model1, model2, model3])
    weights = (1 / errors**k) / np.sum(1 / errors**k)
    return weights

@jit(nopython = True)
def rank_based_weighting(model1, model2, model3):
    errors = np.array([model1, model2, model3])
    ranks = np.argsort(np.argsort(errors)) + 1
    weights = (1 / ranks) / np.sum(1 / ranks)
    return weights

@jit(nopython=True)
def compute_cvar(errors, risk=0.05):
    var_threshold = np.percentile(errors, (risk) * 100)  
    cvar = errors[errors <= var_threshold].mean() 
    return cvar
        