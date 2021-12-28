import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from mean_shift import MeanShift as MeanShift_
import matplotlib.pyplot as plt
from itertools import cycle
# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
X, _ = make_blobs(n_samples=2000, centers=centers, cluster_std=0.6)

# #############################################################################
# Compute clustering with MeanShift

#create array of possible bandwidths
bandwidths = np.linspace(0.1, 1, 10)

# The following bandwidth can be automatically detected using
# bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

for bandwidth in bandwidths:
    ms_ = MeanShift_(bandwidth=bandwidth)
    ms_.fit(X)
    labels_ = ms_.predict(X)
    n_clusters = len(ms_.centroids)
    centroids = ms_.centroids

    plt.figure(1)
    plt.clf()

    colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    for k, col in zip(range(n_clusters), colors):
        my_members = labels_ == k
        cluster_center = centroids[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + ".")
        plt.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=14,
        )
    plt.title("Estimated number of clusters: %d" % n_clusters)
    plt.show()
    print("Bandwidth: %f" % bandwidth)
    print("Estimated number of clusters: %d" % n_clusters)
    print("Cluster centers:\n", centroids)
    print("Labels:\n", labels_)

    print("\n")




