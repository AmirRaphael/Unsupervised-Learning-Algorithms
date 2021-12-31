import time

import numpy as np
from multiprocessing import Pool


def gaussian_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))


def shift(point, data, bandwidth):
    neighbors = data[np.linalg.norm(data - point, axis=1) <= bandwidth]
    if len(neighbors) == 0:
        return point
    else:
        K = np.asarray([gaussian_kernel(point, neighbor, bandwidth) for neighbor in neighbors])
        return np.sum(neighbors * K.reshape(-1, 1), axis=0) / np.sum(K)


class MeanShift:
    def __init__(self, bandwidth=None, max_iter=20, tolerance=1e-6):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.centroids = None
        self.execution_time = None

    def fit(self, data):
        #get start time
        start_time = time.time()
        centroids = np.copy(data)
        p = Pool(10)
        centroids = p.starmap(self.single_center_process, [(centroid, data) for centroid in centroids])
        p.close()
        p.join()
        filtered_centroids = self.filter_centroids(centroids)
        self.centroids = np.asarray(filtered_centroids)

        filtered_centroids = self.filter_centroids(centroids)
        self.centroids = np.asarray(filtered_centroids)
        end_time = time.time()
        self.execution_time = end_time - start_time

    def single_center_process(self, centroid, data):
        for i in range(self.max_iter):
            centroid_new = shift(centroid, data, self.bandwidth)
            if np.linalg.norm(centroid_new - centroid) < self.tolerance:
                break
            centroid = centroid_new
        return centroid

    def predict(self, data):
        distances = np.linalg.norm(self.centroids[:, np.newaxis, :] - data[np.newaxis, :, :], axis=-1)
        return np.argmin(distances, axis=0)

    def filter_centroids(self, centroids):
        filtered_centroids = []
        for centroid in centroids:
            should_add = True
            for c in filtered_centroids:
                if np.linalg.norm(c - centroid) < self.bandwidth:
                    should_add = False
                    break
            if should_add:
                filtered_centroids.append(centroid)
        return filtered_centroids
