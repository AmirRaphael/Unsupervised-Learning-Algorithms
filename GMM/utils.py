import numpy as np


def get_random_spd(n):
    x = np.random.normal(0, 1, size=(n, n))
    return np.dot(x, x.transpose())