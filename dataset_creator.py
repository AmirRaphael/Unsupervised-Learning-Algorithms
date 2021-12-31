import numpy as np
from sklearn import datasets
from sklearn.datasets import make_blobs

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1500


def get_circles():
    return datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)


def get_moons():
    return datasets.make_moons(n_samples=n_samples, noise=0.05)


def get_blobs():
    return make_blobs(n_samples=n_samples, random_state=0, n_features=2)


# Anisotropicly distributed data
def get_aniso():
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    return X_aniso, y


def get_varied():
    # Data is blobs with varied properties
    return datasets.make_blobs(n_samples=n_samples,
                               cluster_std=[1.0, 2.5, 0.5],
                               random_state=170)