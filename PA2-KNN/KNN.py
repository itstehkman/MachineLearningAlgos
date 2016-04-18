from sample import sample, getMatrix
import heapq

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

class KNearestNeighbors:

	#func-names starting with __ are treated as private
	def __init__(self, k):
		''' Initialize and set k'''
		self.k = k

	def fit(self, X, Y):
		''' Fit our KNN to the data with X as the features and 
		Y as the target column'''
		self.X = X
		self.Y = Y

	def predict(self, X_predict):
		''' Fit our KNN to the data with X as the features and 
		Y as the target column. Outputs a vector of predictions'''
		
		preds = np.empty(len(X_predict))
		i = 0
		#for each vector in x_pred, find the euc dist to all v in X
		for row in X_predict:
			norms = LA.norm(self.X - row, axis=1)   #l2 norms of each row
			#pick k top norms
			kIndices = heapq.nsmallest(self.k, xrange(len(norms)), norms.take)
			categories = self.Y[kIndices].astype(int)
			preds[i] = np.bincount(categories).argmax() #get the mode
			i += 1

		self.preds = preds
		return preds

	def score(self, Y_actual):
		return 100.0*len(self.preds[self.preds == Y_actual]) \
			/ len(self.preds)


def zscale(matrix):
	''' Z-Scale the matrix in-place '''
	nCols = len(matrix[0,:])
	
	for i in xrange(nCols):
		sd = np.std(matrix[:,i])
		mn = np.std(matrix[:,i])
		matrix[:,i] -= mn
		matrix[:,i] /= sd

def trainAndGetBestK(x_train, x_test, y_train, y_test):
	'''Trains KNN for the given data and calculates the best k from the
	following values: [1,3,5,7,9]'''
	k_vals = [1,3,5,7,9]
	scores = {}
	for k in k_vals:
		knn = KNearestNeighbors(k)
		knn.fit(x_train, y_train)
		knn.predict(x_test)
		scores[k] = knn.score(y_test)

	k = max(scores, key=lambda i: scores[i])
	return k, scores[k]