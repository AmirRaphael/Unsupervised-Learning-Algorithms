import numpy as np
import matplotlib.pyplot as plt
from gmm import GMM
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


def main():
    # Generate data
    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X_aniso = np.dot(X, transformation)
    #Fit GMM
    gmm = GMM(X=X_aniso, K=3)
    gmm.fit()
    preds = gmm.predict()
    # fit kmeans
    # kmeans = KMeans(n_clusters=3, random_state=random_state)
    # kmeans.fit(X_aniso)
    # preds = kmeans.predict(X_aniso)
    # Plot
    plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=preds)
    plt.title('GMM - Ours')

    plt.show()

if __name__ == '__main__' :
    main()

