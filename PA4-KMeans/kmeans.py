import numpy as np
from numpy import linalg as LA

def append_ones(m):
    return np.concatenate((m, np.ones((m.shape[0], 1))), axis=1)

class KMeans:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X):
        """
        Fit the data with k_clusters by choosing random points. Input
        must be numpy array. Sets the WCSS and centroids fields.
        """
        self.X = X
        idx = np.random.randint(len(X), size=self.k)
        centroids = X[idx]  #index = centroid number
        k_for_data = [-1]*len(X)  #index = index of value in X
        clusters = {}  #mapping cluster numbers to their indices of X
        means = [-1]*self.k  #index = cluster number
        stds = [-1]*self.k  #index = cluster number
        changed = True

        while changed:
            changed = False
            clusters = {}  #must recompute clusters each time

            #update clusters
            for i in range(len(X)):
                min_dist = LA.norm(X[i] - centroids[0])
                min_k = 0
                for j in range(1, self.k):
                    dist = LA.norm(X[i] - centroids[j])
                    if dist < min_dist:
                        min_dist = dist
                        min_k = j
                k_for_data[i] = min_k
                if min_k not in clusters.keys():
                    clusters[min_k] = set()
                clusters[min_k].add(i)

            #update centroids
            for k in range(self.k):
                if k not in clusters.keys():  #what if nothing assigned to centroid
                    continue
                new_c = np.mean(X[list(clusters[k])],axis=0)
                if not np.array_equal(centroids[k],new_c):
                    centroids[k] = new_c
                    changed = True  #break when centroids converge

        print "k=", self.k
        for k in range(self.k):
            means[k] = np.mean(X[list(clusters[k])],axis=0)
            stds[k] = np.std(X[list(clusters[k])],axis=0)
            print "cluster#", k
            print "mean\n", means[k]
            print "std\n", stds[k]

        WCSS = 0
        for i in range(len(X)):
            WCSS += LA.norm(X[i] - centroids[k_for_data[i]])

        self.centroids = centroids
        self.WCSS = WCSS
        self.clusters = clusters
        self.means = means
        self.stds = stds

    def train_clusters(self, Y_train, X_test, Y_test, zscale=False):
        """
        Performs linear regression on each cluster and returns the rmse
        for each cluster.
        """
        #assign clusters for X_test_zscaled
        test_clusters = {}
        k_for_data = {}
        for i in range(len(X_test)):
            min_dist = LA.norm(X_test[i] - self.centroids[0])
            min_k = 0
            for j in range(1, self.k):
                dist = LA.norm(X_test[i] - self.centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    min_k = j
            k_for_data[i] = min_k
            if min_k not in test_clusters.keys():
                test_clusters[min_k] = set()
            test_clusters[min_k].add(i)

        RMSE = 0
        #train and evaluate each cluster
        beta = {}
        for k in self.clusters.keys():
            if k not in self.clusters.keys() or k not in test_clusters.keys():
                continue #the dataset doesn't include both clusters

            train_idx = list(self.clusters[k])
            test_idx = list(test_clusters[k])
            X = self.X[train_idx]
            Y = Y_train[train_idx]
            b, _, _, _ = np.linalg.lstsq(X, Y)
            beta[k] = b

        for i in range(len(X_test)):
            b = beta[k_for_data[i]]
            X = X_test[i]
            Y = Y_test[i]
            RMSE += (np.dot(X, b) - Y) ** 2
        RMSE = np.sqrt(RMSE/len(X_test))

        print "k=", self.k, 'RMSE-abalone', RMSE
        self.RMSE = RMSE
