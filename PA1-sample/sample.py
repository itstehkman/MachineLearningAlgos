import numpy as np
import matplotlib.pyplot as plt
import random

'''
Given a csv filename, returns np array of zeros
with same length as the the # of rows in the csv
'''
def getArray(filename):
	with open(filename) as file:
		row_count = sum(1 for row in file)
		return np.zeros(row_count, dtype='i')

'''
Given a numpy array and percentage, increment the element
of the chosen elements in the array (modify in-place)
'''
def sample(nparray, percent):
	length = len(nparray)
	n_n = float(int(length*percent)) #number needed
	n_r = length #number remaining
	for e in np.nditer(nparray,op_flags=['readwrite']):
		if random.random() <= n_n/n_r :
			e[...] += 1
			n_n -= 1
		n_r -= 1

'''
Given an array of zeroes, do #runs samples and modify x_means,
x_stds, and y_runs accordingly. Clear the array to 0s when done.
'''
def test(array, percent, runs, y_means, y_stds, x_runs):
	for i in xrange(runs):
		sample(array, percent)

	#normalize
	y_means.append( np.mean(array) / runs) 
	y_stds.append(np.std(array) / runs)

	x_runs.append(runs)
	array.fill(0) #set all elems to 0

'''
Given array, run several tests on the array and plot the mean
and SD for each test size.
'''
def runTests(array):
	y_means = []
	y_stds = []
	x_runs = []

	for n in [10, 100, 1000, 10000, 100000]:
		test(array, .1, n, y_means, y_stds, x_runs)

	#plot means
	plt.subplot(211) #plot in (2,1) grid. figure 1
	plt.plot(x_runs, y_means)
	plt.xlabel("# runs")
	plt.ylabel("mean")

	#plot stds
	plt.subplot(212) # plot in (2,1) grid. figure #2
	plt.plot(x_runs, y_stds)
	plt.xlabel("# runs")
	plt.ylabel("Std. Dev")

	plt.show()

###############

random.seed(0) #only seed once
array = getArray('data/abalone.data')
runTests(array)
