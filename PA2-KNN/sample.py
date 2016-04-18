import numpy as np
import matplotlib.pyplot as plt
import random

'''
Given a csv filename, returns a numpy matrix of the data

@param {dict} converters - dict where key is the column number, and value is
the function to convert it. This is useful for string values
'''
def getMatrix(filename, converters=None):
	matrix = np.loadtxt(open(filename,"rb"), delimiter=",", 
		converters=converters)
	return matrix


def sampleHelper(matrix, testPercent):
	'''
	Given a numpy matrix and percentage, return a test/train split 
	(tuple) of that percentage
	'''
	length = len(matrix)
	n_n = float(int(length*testPercent)) #number needed
	n_r = length #number remaining

	nCols = len(matrix[0, 0:])
	test = np.empty([n_n, nCols])
	train = np.empty([length-n_n, nCols])

	tsIdx = 0
	trIdx = 0
	for i in xrange(length):
		if random.random() <= n_n/n_r :
			# test.append(matrix[i])
			test[tsIdx] = matrix[i]
			n_n -= 1
			tsIdx +=1
		else:
			train[trIdx] = matrix[i]
			trIdx +=1
		n_r -= 1

	return test, train


def sample(matrix, seed=0, testPercent=.1):
	'''
	Given a numpy matrix, RNG seed, and percentage, return a tuple
	of 4 matrices:
	x_train, x_test, y_train, y_test
	'''
	random.seed(seed)
	test,train = sampleHelper(matrix, testPercent)

	x_train = train[:,0:len(train[0])-1]   #everything but targ col
	y_train = train[:,len(train[0])-1]     #only targ column

	x_test = test[:,0:len(test[0])-1]  
	y_test = test[:,len(test[0])-1]  

	return x_train, x_test, y_train, y_test

def confusion_matrix(preds, y_test):
	'''Get the confusion matrix for some predictions and some target'''
	cats = np.unique(y_test)
	size = len(cats)
	cm = np.empty((size,size))

	#i is row, j is col. so rows = true, cols = predicted
	i = 0
	for e_i in cats:
		j = 0
		for e_j in cats:
			n = np.sum( np.logical_and(y_test == e_i, preds == e_j) )
			cm[i,j] = n
			j += 1
		i += 1

	return cm

def plot_confusion_matrix(cm, categories):
	'''
	CREDITS TO SCIKIT-LEARN: 
	http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

	Given a 2d confusion matrix and its categories plot it.
	Need to run plt.show() after calling this to actually render'''
	plt.figure(figsize=(12,8))
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.colorbar()
	tick_marks = np.arange(len(categories))
	plt.xticks(tick_marks, categories)
	plt.yticks(tick_marks, categories)
	plt.ylabel('Actual')
	plt.xlabel('Predicted')