import csv
import numpy as np
import matplotlib.pyplot as plt

from kmeans import KMeans

def append_ones(m):
    return np.concatenate((m, np.ones((m.shape[0], 1))), axis=1)

# load data
with open('abalone.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = []
    for row in reader:
        s = row[0]
        sex = [1,0,0] if s=='F' else ([0,1,0] if s=='M' else [0,0,1])
        data.append(sex + [float(r) for r in row[1:]])
    data = np.array(data)

# create train (90%) and test set (10%)
np.random.seed(0)
np.random.shuffle(data)
train_num = int(data.shape[0] * 0.9)
X_train = data[:train_num,:-1]
Y_train = data[:train_num,-1]
X_test  = data[train_num:,:-1]
Y_test  = data[train_num:,-1]

###########################################
# z-scale
X_means = np.mean(X_train, axis=0)
X_stds = np.std(X_train, axis=0)
X_train = (X_train - X_means) / X_stds
X_test = (X_test - X_means) / X_stds

# append ones
X_train = append_ones(X_train)
X_test = append_ones(X_test)

# data = np.arange(1000).reshape((50,20)).astype(np.dtype('f4'))
# np.random.shuffle(data)
# # print data
# train_num = int(data.shape[0] * 0.9)
# X_train = data[:train_num,:-1]
# # print data
# Y_train = data[:train_num,-1]
# X_test  = data[train_num:,:-1]
# Y_test  = data[train_num:,-1]
# e = KMeans(k=4)
# e.fit(X_train)
# e.train_clusters(Y_train, X_test, Y_test)
# print e.RMSE

#########
k_vals = [4]
estimators = [KMeans(k=k) for k in k_vals]
map(lambda e: e.fit(X_train), estimators)
# print X_train
WCSS_list = [e.WCSS for e in estimators]
print WCSS_list

map(lambda e: e.train_clusters(Y_train, X_test, Y_test), estimators)
RMSE_list = [e.RMSE for e in estimators]

# plt.plot(k_vals, RMSE_list)
# plt.show()
