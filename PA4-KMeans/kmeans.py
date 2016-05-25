import numpy as np
from numpy import linalg as LA


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
        changed = True

        while changed:
            changed = False
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


        WCSS = 0
        for i in range(len(X)):
            WCSS += LA.norm(X[i] - centroids[k_for_data[i]])
        self.centroids = centroids
        self.WCSS = WCSS
        self.clusters = clusters

    def train_clusters(self, Y_train, X_test, Y_test):
        """
        Performs linear regression on each cluster and returns the rmse
        for each cluster.
        """
        #assign clusters for X_test
        test_clusters = {}
        for i in range(len(X_test)):
            min_dist = LA.norm(X_test[i] - self.centroids[0])
            min_k = 0
            for j in range(1, self.k):
                dist = LA.norm(X_test[i] - self.centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    min_k = j
            if min_k not in test_clusters.keys():
                test_clusters[min_k] = set()
            test_clusters[min_k].add(i)

        RMSE = 0
        #train and evaluate each cluster
        for k in self.clusters.keys():
            if k not in self.clusters.keys() or k not in test_clusters.keys():
                continue #the dataset doesn't include both clusters

            train_idx = list(self.clusters[k])
            test_idx = list(test_clusters[k])
            X = self.X[train_idx]
            Y = Y_train[train_idx]
            X2 = X_test[test_idx]
            Y2 = Y_test[test_idx]
            print X
            beta, _, _, _ = np.linalg.lstsq(X, Y)
            # root mean square error
            print np.dot(X2, beta)
            print Y2
            RMSE += np.sqrt(np.mean((np.dot(X2, beta) - Y2) ** 2))
        print "k=", self.k, 'RMSE-abalone', RMSE
        self.RMSE = RMSE
