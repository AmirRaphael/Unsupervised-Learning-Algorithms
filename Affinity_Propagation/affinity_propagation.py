import numpy as np
from sklearn.metrics import euclidean_distances


# TODO: refactor implementation
def compute_squared_euclidean_distances(M: np.ndarray) -> np.ndarray:
    """
    Considering M as a collection of vectors (x_1,...,x_n),
    compute the n x n matrix D of all pairwise distances between the vectors.
    :param M: ndarray of shape (n, m) - matrix
    :return: distances matrix D: ndarray of shape (n, n) - d_ij = ||x_i - x_j||^2
    We will use the alternative definition: D_ij = ||x_i||^2 - 2(x_i^T * x_j) + ||x_j||^2
    """
    # Get dimensions of M
    n, m = M.shape
    # Compute the Gram matrix
    G = M @ M.T
    # Compute H where: H_ij == G_jj == ||x_j||^2 and H_ji == G_ii == ||x_i||^2
    H = np.tile(np.diag(G), (n, 1))
    # Compute D
    D = H - (2 * G) + H.T
    return D


# TODO: refactor implementation
def compute_responsibility(
        S: np.ndarray,
        A: np.ndarray,
        R: np.ndarray,
        damping_factor: float
) -> np.ndarray:
    """
    Compute the responsibilities.
    :param S: ndarray of shape (n_samples, n_samples) - similarity matrix
    :param A: ndarray of shape (n_samples, n_samples) - availability matrix
    :param R: ndarray of shape (n_samples, n_samples) - responsibility matrix
    :param damping_factor: float in range (0.5, 1.0) - the extent to which the current value is maintained relative to incoming values
    :return: new responsibility matrix: ndarray of shape (n_samples, n_samples)
    the responsibility of a sample k to be the exemplar of sample i is computed by:
    r(i,k) = s(i,k) - max{a(i,k') + s(i,k')} [k' != k]
    """
    # Compute a(i,k') + s(i,k') & ignore the diagonal of A + S (where k' = k)
    T = A + S
    np.fill_diagonal(T, -np.inf)

    # Get row indices
    row_indices = np.arange(T.shape[0])

    # Compute max{a(i,k') + s(i,k')} (Get maximum value of each row)
    max_col_indices = np.argmax(T, axis=1)
    row_max = T[row_indices, max_col_indices]

    # Assign -inf to previous maximum in order to get secondary maximum
    T[row_indices, max_col_indices] = -np.inf

    # Get secondary maximum value of each row
    secondary_max_col_indices = np.argmax(T, axis=1)
    row_secondary_max = T[row_indices, secondary_max_col_indices]

    # Create matrix of max{a(i,k') + s(i,k')}
    max_AS = np.zeros_like(S) + row_max.reshape(-1, 1)

    # Modify values of those indices where there were maxima because k' != k
    max_AS[row_indices, max_col_indices] = row_secondary_max

    # Compute s(i,k) - max{a(i,k') + s(i,k')} (new responsibilities)
    new_R = S - max_AS

    # In order to avoid numerical oscillations when updating the messages, the damping factor is used in the following way:
    # r_new(i,k) = (damping_factor * r_cur(i,k)) + ((1 - damping_factor) * r_new(i,k))
    return (damping_factor * R) + ((1 - damping_factor) * new_R)


# TODO: refactor implementation
def compute_availability(
        A: np.ndarray,
        R: np.ndarray,
        damping_factor: float
) -> np.ndarray:
    """
    Compute the availabilities.
    :param A: ndarray of shape (n_samples, n_samples) - availability matrix
    :param R: ndarray of shape (n_samples, n_samples) - responsibility matrix
    :param damping_factor: float in range (0.5, 1.0) - the extent to which the current value is maintained relative to incoming values
    :return: new availability matrix: ndarray of shape (n_samples, n_samples)
    the availability of a sample k to be the exemplar of sample i is computed by:
    [for i != k] => a(i,k) = min{0, r(k,k) + sum(max{0, r(i',k)})} [i' != i,k]
    [for i = k] => a(k,k) = sum(max{0, r(i',k)}) [i' != k]
    """
    # copy the responsibility matrix
    R = R.copy()
    diag_R = np.diag(R).copy()

    # Fill diagonal with 0
    np.fill_diagonal(R, 0)

    # Replace all negative responsibilities with 0
    R = np.where(R < 0, 0, R)

    # Compute availabilities

    # First, make matrix with column sum in each cell of that column
    # Note: This is still without diagonal / negative values
    # Compute sum(max{0, r(i',k)})} [i' != i,k]
    sum_R = np.sum(R, axis=0)

    # Compute a(i,k) = min{0, r(k,k) + sum(max{0, r(i',k)})} [i' != i,k]
    new_A = np.minimum(0, (diag_R + sum_R - R))

    # Compute self-availabilities
    # Note that diagonal in R is 0
    # Compute a(k,k) = sum(max{0, r(i',k)}) [i' != k]
    np.fill_diagonal(new_A, np.sum(R, axis=0))

    # In order to avoid numerical oscillations when updating the messages, the damping factor is used in the following way:
    # a_new(i,k) = (damping_factor * a_cur(i,k)) + ((1 - damping_factor) * a_new(i,k))
    return (damping_factor * A) + ((1 - damping_factor) * new_A)


# TODO: refactor implementation
def affinity_propagation(
        S: np.ndarray,
        convergence_iter=15,
        max_iter=200,
        damping_factor=0.5
):
    """
    Run the affinity propagation clustering algorithm.
    :param S: ndarray of shape (n_samples, n_samples) - similarity matrix
    :param convergence_iter: int - number of iterations without change in the number of clusters which stops the algorithm.
    :param max_iter: int - maximum number of iterations.
    :param damping_factor: float in range (0.5, 1.0) - the extent to which the current value is maintained relative to incoming values
    :return:
    exemplar_indices: array - index of elexmplars (cluster centers).
    labels: array of size (n_samples) - cluster labels for each point.
    n_iter: int - total number of iterations until convergence.
    """
    n_samples = S.shape[0]

    # Initialize messages
    A = np.zeros((n_samples, n_samples)) # availability matrix
    R = np.zeros((n_samples, n_samples)) # responsibility matrix

    # TODO: check if this is required
    # Remove degeneracies to avoid oscillating
    # S = S + 1e-12 * np.random.normal(size=A.shape) * (np.max(S) - np.min(S))

    n_iter = 0
    convergence_count = 0

    for iteration in range(max_iter):
        n_iter = iteration + 1
        curr_sum_RA = R + A
        curr_labels = np.argmax(curr_sum_RA, axis=1)
        R = compute_responsibility(S, A, R, damping_factor)
        A = compute_availability(A, R, damping_factor)
        new_sum_RA = R + A
        new_labels = np.argmax(new_sum_RA, axis=1)
        # check if clusters changed
        if np.all(curr_labels == new_labels):
            convergence_count += 1
            if convergence_count >= convergence_iter:
                break
        else:
            convergence_count = 0

    sum_RA = R + A
    labels = np.argmax(sum_RA, axis=1)
    exemplar_indices = np.unique(labels)

    # rearrange exemplar indices and labels to be values from [0,1,...,k]
    replace = dict(zip(exemplar_indices, range(len(exemplar_indices))))
    mp = np.arange(0, max(labels) + 1)
    mp[list(replace.keys())] = list(replace.values())
    labels = mp[labels]

    return exemplar_indices, labels, n_iter


class AffinityPropagation:
    def __init__(
            self,
            convergence_iter=15,
            max_iter=200,
            damping_factor=0.5,
            preference="median"
    ):
        self.convergence_iter = convergence_iter
        self.max_iter = max_iter
        self.damping_factor = damping_factor
        self.preference = preference
        self.similarity_matrix = None
        self.exemplar_indices = None
        self.labels = None
        self.n_iter = None

    def fit(self, X: np.ndarray):
        """
        Fit the clustering from features.
        :param X: array of shape (n_samples, n_features) - data
        :return: self - the instance of this AffinityPropagation class
        """
        # The similarity of 2 data points is defined as the negative squared euclidean distance between them
        self.similarity_matrix = -compute_squared_euclidean_distances(X)

        if self.preference == "median":
            # By default preference is defined to be the median similarity of all pairs of inputs
            preference = np.median(self.similarity_matrix)
            np.fill_diagonal(self.similarity_matrix, preference)
        elif self.preference == "min":
            # Setting the preference to the minimum similarity of all pairs of inputs will result in fewer clusters
            preference = np.min(self.similarity_matrix) - 1
            np.fill_diagonal(self.similarity_matrix, preference)
        else:
            raise ValueError("preference must be 'median' or 'min'")

        (
            self.exemplar_indices,
            self.labels,
            self.n_iter
        ) = affinity_propagation(
            S=self.similarity_matrix,
            convergence_iter=self.convergence_iter,
            max_iter=self.max_iter,
            damping_factor=self.damping_factor
        )

        return self
