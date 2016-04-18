import numpy as np
import matplotlib.pyplot as plt

from sample import sample, getMatrix, confusion_matrix, plot_confusion_matrix
from KNN import zscale, KNearestNeighbors, trainAndGetBestK

####################### ABALONE #######################
#Hard-coded converter function for the Abalone dataset
def convertAbalone(e):
	if e=="M":
		return 0
	elif e=="F":
		return 1
	elif e=="I":
		return 2

converters = {0: convertAbalone}
data = getMatrix('data/abalone.data', converters)

#get masks for M, F, and I individually
genders = data[:,0]
M = (genders == 0).astype(int)
F = (genders == 1).astype(int)
I = (genders == 2).astype(int)

#apply the masks to get individual cols for M, F, I
data[:,0] = M
data = np.insert(data, 0, F, axis=1)
data = np.insert(data, 0, I, axis=1)

x_train, x_test, y_train, y_test = sample(data)

y_train = y_train.astype(int)
y_test = y_test.astype(int)
zscale(x_train)	
zscale(x_test)
#now try KNN

#UNCOMMENT TO GET BEST K, WHICH IS K=5, SCORE=22.5%
# k,score = trainAndGetBestK(x_train, x_test, y_train, y_test)
# print k

knn = KNearestNeighbors(5)
knn.fit(x_train, y_train)
preds = knn.predict(x_test)

cm = confusion_matrix(preds, y_test)
cats = np.unique(y_train)

plot_confusion_matrix(cm, cats)
plt.title('Abalone, k = 5, score = 22.5%')
plt.show()