import numpy as np
import matplotlib.pyplot as plt

from sample import sample, getMatrix, confusion_matrix, plot_confusion_matrix
from KNN import zscale, KNearestNeighbors, trainAndGetBestK

############ 3 PERCENT SEPERABLE ##############
data = getMatrix('data/Seperable.csv')
x_train, x_test, y_train, y_test = sample(data)

zscale(x_train)	
zscale(x_test)

# UNCOMMENT TO FIND BEST K, WHICH IS K = 1, SCORE = 62.6%
# k,score = trainAndGetBestK(x_train, x_test, y_train, y_test)
# print k, score

knn = KNearestNeighbors(1)
knn.fit(x_train, y_train)
preds = knn.predict(x_test)

cm = confusion_matrix(preds, y_test)
cats = np.unique(y_train)

plot_confusion_matrix(cm, cats)
plt.title('seperable, k = 1, score = 62.6%')
plt.show()