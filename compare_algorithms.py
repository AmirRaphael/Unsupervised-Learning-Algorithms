import matplotlib.pyplot as plt
from Affinity_Propagation.affinity_propagation import AffinityPropagation as _AffinityPropagation
from Mean_Shift.mean_shift import MeanShift as _MeanShift
from GMM.gmm import GMM as _GMM
from dataset_creator import get_blobs, get_moons, get_circles, get_aniso, get_varied


def main():
    circle_data = (get_circles(), {"damping": 0.77, "bandwidth": 0.9, "n_gaussians": 2})
    moons_data = (get_moons(), {"damping": 0.75, "bandwidth": 0.9, "n_gaussians": 2})
    varied_data = (get_varied(), {"damping": 0.9, "bandwidth": 1.1, "n_gaussians": 3})
    aniso_data = (get_aniso(), {"damping": 0.9, "bandwidth": 1.1, "n_gaussians": 3})
    blobs_data = (get_blobs(), {"damping": 0.9, "bandwidth": 1.1, "n_gaussians": 3})

    # put all data in list
    datasets = [blobs_data, circle_data, varied_data, aniso_data, moons_data]

    for (dataset, params) in datasets:
        X, y = dataset
        ap = _AffinityPropagation(preference="min", damping_factor=params["damping"]).fit(X)
        labels_ap = ap.labels
        centers_ap = ap.cluster_centers
        ms = _MeanShift(bandwidth=params["bandwidth"])
        ms.fit(X)
        labels_ms = ms.predict(X)
        centers_ms = ms.centroids
        gmm = _GMM(X=X, K=params["n_gaussians"])
        gmm.fit()
        labels_gmm = gmm.predict()
        means_gmm = gmm.means

        # plot results of each algorithm with centroids
        plt.figure(figsize=(10, 10))
        plt.subplot(221)
        plt.scatter(X[:, 0], X[:, 1], c=labels_ap, s=50, cmap='viridis')
        plt.scatter(centers_ap[:, 0], centers_ap[:, 1], c='black', s=200, alpha=0.5)
        plt.title('Affinity Propagation')
        plt.subplot(222)
        plt.scatter(X[:, 0], X[:, 1], c=labels_ms, s=50, cmap='viridis')
        plt.scatter(centers_ms[:, 0], centers_ms[:, 1], c='black', s=200, alpha=0.5)
        plt.title('Mean Shift')
        plt.subplot(223)
        plt.scatter(X[:, 0], X[:, 1], c=labels_gmm, s=50, cmap='viridis')
        plt.scatter(means_gmm[:, 0], means_gmm[:, 1], c='black', s=200, alpha=0.5)
        plt.title('GMM')
        plt.show()


if __name__ == '__main__':
    main()