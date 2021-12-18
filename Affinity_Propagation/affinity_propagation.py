import numpy as np


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


def update_responsibilities(
        S: np.ndarray,
        A: np.ndarray,
        R: np.ndarray,
        damping_factor: float
) -> np.ndarray:
    """
    Update the responsibilities.
    :param S: ndarray of shape (n_samples, n_samples) - similarity matrix
    :param A: ndarray of shape (n_samples, n_samples) - availability matrix
    :param R: ndarray of shape (n_samples, n_samples) - responsibility matrix
    :param damping_factor: float in range (0.5, 1.0)
    :return: new responsibility matrix: ndarray of shape (n_samples, n_samples)
    the responsibility of a sample k to be the exemplar of sample i is computed by:
    r(i,k) = s(i,k) - max{a(i,k') + s(i,k')} [k' != k]
    """
    n_samples = S.shape[0]
    indices = np.arange(n_samples)

    # Compute a(i,k') + s(i,k')
    T = A + S

    # Compute max{a(i,k') + s(i,k')}
    max_indices = np.argmax(T, axis=1)
    max_values = T[indices, max_indices]

    # Compute max{a(i,k') + s(i,k')} again in order to get secondary max values
    T[indices, max_indices] = -np.inf  # Ignore primary max values
    secondary_max_indices = np.argmax(T, axis=1)
    secondary_max_values = T[indices, secondary_max_indices]

    # Build matrix of max{a(i,k') + s(i,k')} where k' != k
    M = np.zeros((n_samples, n_samples)) + max_values.reshape(-1, 1)
    M[indices, max_indices] = secondary_max_values

    # Compute s(i,k) - max{a(i,k') + s(i,k')} [k' != k] (new responsibilities)
    new_R = S - M

    # In order to avoid numerical oscillations when updating the messages, the damping factor is used in the following way:
    # r_new(i,k) = (damping_factor * r_cur(i,k)) + ((1 - damping_factor) * r_new(i,k))
    return (damping_factor * R) + ((1 - damping_factor) * new_R)


def update_availabilities(
        A: np.ndarray,
        R: np.ndarray,
        damping_factor: float
) -> np.ndarray:
    """
    Update the availabilities.
    :param A: ndarray of shape (n_samples, n_samples) - availability matrix
    :param R: ndarray of shape (n_samples, n_samples) - responsibility matrix
    :param damping_factor: float in range (0.5, 1.0)
    :return: new availability matrix: ndarray of shape (n_samples, n_samples)
    the availability of a sample k to be the exemplar of sample i is computed by:
    [for i != k] => a(i,k) = min{0, r(k,k) + sum(max{0, r(i',k)}) [i' != i,k]
    [for i = k] => a(k,k) = sum(max{0, r(i',k)}) [i' != k]
    """
    n_samples = A.shape[0]

    # Create a temp copy of the responsibility matrix and it's diagonal
    T = R.copy()  # T[i,k] = r(i,k)
    diag_T = np.diag(T).copy()  # diag_T[k] = r(k,k)

    # Compute sum(max{0, r(i',k)}) [i' != k]
    np.fill_diagonal(T, 0)  # Ignore values on the diagonal because we need i' != k
    T[T < 0] = 0  # Ignore negative values because we need max{0, r(i',k)}
    sum_T = np.sum(T, axis=0)  # sum_T[k] = sum(T[i',k]) for all i' != k

    # Compute a(i,k) = min{0, r(k,k) + sum(max{0, r(i',k)}) [i' != i,k]
    new_A = np.zeros((n_samples, n_samples)) + (diag_T + sum_T)  # new_A[i,k] = r(k,k) + sum(max{0, r(i',k)}) [i' != k]
    new_A -= T  # Make correction since we need to use sum(max{0, r(i',k)}) where i' != i
    new_A = np.minimum(0, new_A)  # Replace values greater than zero

    # Compute a(k,k) = sum(max{0, r(i',k)}) [i' != k]
    np.fill_diagonal(new_A, sum_T)

    # In order to avoid numerical oscillations when updating the messages, the damping factor is used in the following way:
    # a_new(i,k) = (damping_factor * a_cur(i,k)) + ((1 - damping_factor) * a_new(i,k))
    return (damping_factor * A) + ((1 - damping_factor) * new_A)


def affinity_propagation(
        S: np.ndarray,
        convergence_iter=15,
        max_iter=200,
        damping_factor=0.5
):
    """
    Run the affinity propagation clustering algorithm.
    :param S: ndarray of shape (n_samples, n_samples) - similarity matrix
    :param convergence_iter: int - number of iterations without change in the clusters which stops the algorithm.
    :param max_iter: int - maximum number of iterations.
    :param damping_factor: float in range (0.5, 1.0)
    :return:
    exemplar_indices: array - index of exemplars (cluster centers).
    labels: array of size (n_samples) - cluster labels for each point.
    n_iter: int - total number of iterations until convergence.
    """
    n_samples = S.shape[0]

    # Create message matrices
    A = np.zeros((n_samples, n_samples))  # availability matrix
    R = np.zeros((n_samples, n_samples))  # responsibility matrix

    n_iter = 0
    convergence_count = 0

    for iteration in range(max_iter):
        n_iter = iteration + 1

        # Get current labels for each data-point x_i: (max{r(i,j) + a(i,j)})
        curr_labels = np.argmax(R + A, axis=1)

        # Compute new responsibilities and availabilities
        R = update_responsibilities(S, A, R, damping_factor)
        A = update_availabilities(A, R, damping_factor)

        # Get new labels for each data-point x_i: (max{r(i,j) + a(i,j)})
        new_labels = np.argmax(R + A, axis=1)

        # check if clusters changed
        if np.array_equal(curr_labels, new_labels):
            convergence_count += 1
            if convergence_count >= convergence_iter:
                break
        else:
            convergence_count = 0

    # Get final labels for each data-point x_i: (max{r(i,j) + a(i,j)})
    labels = np.argmax(R + A, axis=1)
    exemplar_indices = np.unique(labels)

    # rearrange labels to be values from [0,1,...,n_clusters]
    labels = np.searchsorted(exemplar_indices, labels)

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
        self.exemplar_indices = None
        self.labels = None
        self.n_iter = None

    def fit(self, X: np.ndarray):
        """
        Find cluster centers (exemplars) for the input data.
        :param X: array of shape (n_samples, n_features) - data
        :return: self - the instance of this AffinityPropagation class
        """
        # The similarity of 2 data points is defined as the negative squared euclidean distance between them
        similarity_matrix = -compute_squared_euclidean_distances(X)

        if self.preference == "median":
            # By default preference is defined to be the median similarity of all pairs of inputs
            preference = np.median(similarity_matrix)
            np.fill_diagonal(similarity_matrix, preference)
        elif self.preference == "min":
            # Setting the preference to the minimum similarity of all pairs of inputs will result in fewer clusters
            preference = np.min(similarity_matrix) - 1
            np.fill_diagonal(similarity_matrix, preference)
        else:
            raise ValueError("preference must be 'median' or 'min'")

        (
            self.exemplar_indices,
            self.labels,
            self.n_iter
        ) = affinity_propagation(
            S=similarity_matrix,
            convergence_iter=self.convergence_iter,
            max_iter=self.max_iter,
            damping_factor=self.damping_factor
        )

        return self
