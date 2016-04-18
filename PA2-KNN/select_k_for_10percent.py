import numpy as np
import matplotlib.pyplot as plt

from sample import sample, getMatrix, confusion_matrix, plot_confusion_matrix
from KNN import zscale, KNearestNeighbors, trainAndGetBestK

############ 10 PERCENT MISCATEGORIZ ##############
data = getMatrix('data/10percent-miscatergorization.csv')
x_train, x_test, y_train, y_test = sample(data)

zscale(x_train)	
zscale(x_test)

# UNCOMMENT TO FIND BEST K, WHICH IS K = 3, SCORE = 64.2%
# k,score = trainAndGetBestK(x_train, x_test, y_train, y_test)
# print k, score

knn = KNearestNeighbors(3)
knn.fit(x_train, y_train)
preds = knn.predict(x_test)

cm = confusion_matrix(preds, y_test)
cats = np.unique(y_test)

plot_confusion_matrix(cm, cats)
plt.title('10percent, k = 3, score = 64.2%')
plt.show()