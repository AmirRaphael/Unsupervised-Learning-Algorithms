import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from Mean_Shift.mean_shift import MeanShift as _MeanShift
from dataset_creator import get_blobs, get_moons, get_circles, get_aniso, get_varied


def main():
    circle_data = get_circles()
    varied_data = get_varied()
    blobs_data = get_blobs()
    aniso_data = get_aniso()
    moons_data = get_moons()

    # put all data in list
    datasets = [circle_data, varied_data, blobs_data, aniso_data, moons_data]

    for bandwidth in [0.7, 0.9, 1.1, 1.3]:
        for dataset in datasets:
            # get data
            X, y = dataset
            # create MeanShift object
            ms1 = _MeanShift(bandwidth=bandwidth)
            # fit the model
            ms1.fit(X)
            # get labels
            label1 = ms1.predict(X)
            # get the cluster centers
            centers1 = ms1.centroids
            ms2 = MeanShift(bandwidth=bandwidth)
            ms2.fit(X)
            label2 = ms2.predict(X)
            centers2 = ms2.cluster_centers_

            # plot results of both models
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.scatter(X[:, 0], X[:, 1], c=label1, s=50, cmap='viridis')
            plt.scatter(centers1[:, 0], centers1[:, 1], c='black', s=200, alpha=0.5)
            plt.title('our MeanShift with bandwidth = {}'.format(bandwidth))
            plt.subplot(1, 2, 2)
            plt.scatter(X[:, 0], X[:, 1], c=label2, s=50, cmap='viridis')
            plt.scatter(centers2[:, 0], centers2[:, 1], c='black', s=200, alpha=0.5)
            plt.title('sklearn MeanShift with bandwidth = {}'.format(bandwidth))
            plt.show()


if __name__ == '__main__':
    main()
