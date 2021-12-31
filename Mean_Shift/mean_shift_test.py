import time

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
            X, y = dataset
            ms1 = _MeanShift(bandwidth=bandwidth)
            ms1.fit(X)
            label1 = ms1.predict(X)
            centers1 = ms1.centroids
            ms2 = MeanShift(bandwidth=bandwidth)
            time2 = time.time()
            ms2.fit(X)
            time2 = time.time() - time2
            label2 = ms2.predict(X)
            centers2 = ms2.cluster_centers_

            #print execution time of both algorithms
            print("Execution time of our MeanShift with bandwidth = {}: {}".format(bandwidth, ms1.execution_time))
            print("Execution time of sklearn's MeanShift with bandwidth = {}: {}".format(bandwidth, time2))

            # plot results of both models
            plt.figure(figsize=(10, 8))
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
