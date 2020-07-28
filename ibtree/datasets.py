import numpy as np


def grid2d(N):
    """
    Create an (N*N, 2) matrix of samples on a grid of [0,1]x[0,1]
    """
    X = np.zeros(((N)*(N), 2))
    X[:, 0] = np.repeat(np.arange(0, 1, 1/N), N)
    X[:, 1] = np.concatenate([np.arange(0, 1, 1/N)]*N)
    return X-X.mean(0)


def circle_y(X):
    y = np.zeros((10000, 2))
    y[np.square(X+0.5).sum(1) > 0.2, 0] = 1
    y[:, 1] = 1 - y.sum(1)
    return y

    
def circle_data():
    X = grid2d(100)
    y = circle_y(X)
    return X, y


def caro_data():
    X = grid2d(100)
    y = np.zeros((10000, 2))
    y[(X[:, 0] < 0) == (X[:, 1] < 0), 0] = 1
    y[:, 1] = 1 - y.sum(1)
    return X, y


def easy_data():
    X = grid2d(100)
    y = np.zeros((10000, 2))
    y[X[:, 0] < 0.1, 0] = 1
    y[X[:, 1] < -0.3, 0] = 1
    y[:, 1] = 1 - y.sum(1)
    return X, y
