import numpy as np


def affinity_propagation(
        similarity_matrix,
        convergence_iter=15,
        max_iter=200,
        damping_factor=0.5,
):
    """
    Run the affinity_propagation clustering algorithm.
    :param similarity_matrix: array of shape (n_samples, n_samples) - describes similarity between points.
    :param convergence_iter: int - number of iterations without change in the number of clusters which stops the algorithm.
    :param max_iter: int - maximum number of iterations.
    :param damping_factor: float in range (0.5, 1.0) - the extent to which the current value is maintained relative to incoming values
    :return:
    cluster_centers_indices: array of shape (n_clusters) - index of clusters centers.
    labels: array of shape (n_samples) - cluster labels for each point.
    n_iter: int - total number of iterations until convergence.
    """
    pass


class AffinityPropagation:
    def __init__(
            self,
            convergence_iter=15,
            max_iter=200,
            damping=0.5,
    ):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter

    def fit(self, X):
        """
        Fit the clustering from features
        :param X: array of shape (n_samples, n_features) - data
        :return: self - the instance of this AffinityPropagation class
        """
        pass
