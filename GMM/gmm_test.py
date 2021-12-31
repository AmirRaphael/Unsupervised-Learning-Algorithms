import numpy as np
import matplotlib.pyplot as plt
from gmm import GMM
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from dataset_creator import get_blobs, get_moons, get_circles, get_aniso, get_varied


def main():
    circle_data = get_circles()
    varied_data = get_varied()
    blobs_data = get_blobs()
    aniso_data = get_aniso()
    moons_data = get_moons()

    # put all data in list
    datasets = [circle_data, varied_data, blobs_data, aniso_data, moons_data]

    for dataset in datasets:
        dataset = dataset[0]
        # create GMM object
        gmm1 = GMM(dataset, 3)
        # fit GMM
        gmm1.fit()

        gmm2 = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
        gmm2.fit(dataset)

        # plot results of both implementations
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.scatter(dataset[:, 0], dataset[:, 1], c=gmm1.predict())
        plt.title('our GMM')
        plt.subplot(1, 2, 2)
        plt.scatter(dataset[:, 0], dataset[:, 1], c=gmm2.predict(dataset))
        plt.title('sklearn GMM')
        plt.show()


if __name__ == '__main__':
    main()
