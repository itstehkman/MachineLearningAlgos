import numpy as np
import matplotlib.pyplot as plt
import random

'''
Given a csv filename, returns a numpy matrix of the data

@param {dict} converters - dict where key is the column number, and value is
the function to convert it. This is useful for string values
'''
def getMatrix(filename, converters):
	# with open(filename) as file:
	# 	row_count = sum(1 for row in file)
	# 	return [0]*row_count
	matrix = np.loadtxt(open(filename,"rb"), delimiter=",", 
		converters=converters)
	return matrix

'''
Given a numpy matrix and percentage, return a test/train split of that percentage
'''
def sampleHelper(matrix, testPercent):
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

'''
Main function. Given a csv filename, RNG seed, and percentage, return a tuple
of 2 matrices:
First, the train numpy matrix. Second the test numpy matrix.
'''
def sample(filename, converters, seed=0, testPercent=.1):
	random.seed(seed)
	matrix = getMatrix(filename, converters)
	test,train = sampleHelper(matrix, testPercent)
	return test, train

#########

'''Hard-coded converter function for the Abalone dataset'''
def convertAbalone(e):
	if e=="M":
		return 0
	elif e=="F":
		return 1
	elif e=="I":
		return 2

converters = {0: convertAbalone}
sample('data/abalone.data', converters)