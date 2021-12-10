import numpy as np


def euclidean_distances_squared(M1, M2):
    """
    Considering the rows of M1 and M2 as vectors, compute the squared distance matrix between each pair of vectors.
    :param M1: array of shape (n_samples_M1, n_features)
    :param M2: array of shape (n_samples_M2, n_features)
    :return: distances: array of shape (n_samples_M1, n_samples_M2)
    """
    # TODO: implement
    pass


def affinity_propagation(
        similarity_matrix,
        convergence_iter=15,
        max_iter=200,
        damping_factor=0.5
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
    # TODO: implement
    n_samples = similarity_matrix.shape[0]

    # Initialize messages
    availability_matrix = np.zeros((n_samples, n_samples))
    responsibility_matrix = np.zeros((n_samples, n_samples))

    pass


class AffinityPropagation:
    def __init__(
            self,
            convergence_iter=15,
            max_iter=200,
            damping_factor=0.5,
    ):
        self.damping_factor = damping_factor
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.similarity_matrix = None
        self.cluster_centers_indices = None
        self.labels = None
        self.n_iter = None

    def fit(self, X):
        """
        Fit the clustering from features.
        :param X: array of shape (n_samples, n_features) - data
        :return: self - the instance of this AffinityPropagation class
        """
        # the similarity of 2 data points is defined as the negative squared euclidean distance between them
        self.similarity_matrix = -euclidean_distances_squared(X, X)

        (
            self.labels,
            self.cluster_centers_indices,
            self.n_iter
        ) = affinity_propagation(
            similarity_matrix=self.similarity_matrix,
            convergence_iter=self.convergence_iter,
            max_iter=self.max_iter,
            damping_factor=self.damping_factor
        )

        return self
