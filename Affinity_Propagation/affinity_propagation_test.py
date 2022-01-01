import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from Affinity_Propagation.affinity_propagation import AffinityPropagation as _AffinityPropagation
from dataset_creator import get_blobs, get_moons, get_circles, get_aniso, get_varied


def main():
    circle_data = (get_circles(), {"damping": 0.77})
    moons_data = (get_moons(), {"damping": 0.75})
    varied_data = (get_varied(), {"damping": 0.9})
    aniso_data = (get_aniso(), {"damping": 0.9})
    blobs_data = (get_blobs(), {"damping": 0.9})

    # put all data in list
    datasets = [blobs_data, circle_data, varied_data, aniso_data, moons_data]

    for pref in ["min", "median"]:
        for (dataset, params) in datasets:
            X, y = dataset
            ap = _AffinityPropagation(preference=pref, damping_factor=params["damping"]).fit(X)
            label1 = ap.labels
            centers1 = ap.cluster_centers
            print('Our AP finished with {} clusters'.format(len(centers1)))
            ap2 = AffinityPropagation(preference=ap.preference_val, damping=params["damping"]).fit(X)
            label2 = ap2.labels_
            centers2 = ap2.cluster_centers_
            print('Sklearn AP finished with {} clusters'.format(len(centers2)))

            # plot results of both models
            plt.figure(figsize=(10, 8))
            plt.subplot(1, 2, 1)
            plt.scatter(X[:, 0], X[:, 1], c=label1, s=50, cmap='viridis')
            plt.scatter(centers1[:, 0], centers1[:, 1], c='black', s=200, alpha=0.5)
            plt.title('our AP with pref = {}'.format(pref))
            plt.subplot(1, 2, 2)
            plt.scatter(X[:, 0], X[:, 1], c=label2, s=50, cmap='viridis')
            plt.scatter(centers2[:, 0], centers2[:, 1], c='black', s=200, alpha=0.5)
            plt.title('sklearn AP with pref = {}'.format(pref))
            plt.show()


if __name__ == '__main__':
    main()