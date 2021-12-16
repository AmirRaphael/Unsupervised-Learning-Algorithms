import numpy as np
from scipy.stats import multivariate_normal

from GMM.utils import get_random_spd


def gaussian(X, mean, cov):
    return multivariate_normal.pdf(X, mean, cov, allow_singular=True)


class GMM:
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.M = X.shape[0]  # number of data points
        self.N = X.shape[1]  # number of features
        self.means = np.zeros((K, self.N))  # means of the K Gaussians
        self.covs = np.zeros((K, self.N, self.N))  # covariance matrices of the K Gaussians
        self.pis = np.zeros(K)  # mixture coefficients
        self.Z = np.zeros((self.M, self.K))  # responsibilities

    def init_random_params(self):
        self.covs = np.asarray([get_random_spd(self.N) for _ in range(self.K)])
        self.means = np.random.randn(self.K, self.N)
        self.pis = np.random.rand(self.K)
        self.pis /= np.sum(self.pis)  # normalize so that the sum is 1

    def e_step(self):
        for k in range(self.K):
            self.Z[:, k] = self.pis[k] * gaussian(self.X, self.means[k], self.covs[k])
        self.Z /= np.sum(self.Z, axis=1, keepdims=True)

    def m_step(self):
        Nk = np.sum(self.Z, axis=0)
        for k in range(self.K):
            z_k = self.Z[:, k].reshape(-1, 1)
            self.means[k] = np.sum(z_k * self.X, axis=0) / Nk[k]
            self.covs[k] = np.dot((z_k * (self.X - self.means[k])).T, (self.X - self.means[k])) / Nk[k]
            self.pis[k] = Nk[k] / self.M

    def fit(self, max_iter=100, tol=1e-3):
        self.init_random_params()
        for i in range(max_iter):
            self.e_step()
            self.m_step()

    def predict(self):
        return np.argmax(self.Z, axis=1)




