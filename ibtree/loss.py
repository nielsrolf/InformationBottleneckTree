from scipy.special import xlogy
import numpy as np


def entropy(y):
    """Return the empirical entropy of samples y of 
    a categorical distribution

    Arguments:
        y: np.array (N, C) , categorical labels
    Returns:
        H: float
    """
    if len(y) == 0:
        return 0
    py = y.mean(0)
    h = -np.sum(xlogy(py, py))
    return h


def categorical_IB(beta):
    def J(y):
        if len(y) == 0:
            return 0
        return beta*entropy(y) - np.log(len(y))
    return J