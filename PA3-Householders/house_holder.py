import csv
import sys

import numpy as np
from numpy import linalg as LA
from scipy.linalg import block_diag

sys.path.append("..")  #a hack so that the path can see the essentials folder
import essentials.sampler as utils  #my sampler

def qr_decompose(X):
	"""
	QR-Decompose matrix X using householder's reflection for each column of X.
	Returns Q,R
	"""
	f4 = np.dtype("f4")
	n,d = X.shape
	R = X.astype(f4)  #32 bit fp
	Q_acc = np.identity(n)

	for i in xrange(d):
		if i >= n-1:
			break
		z = R[i:n,i]
		z = np.array([z]).T  #turn 1d array into column vector
		e1 = np.zeros(shape=z.shape, dtype=int)
		e1[0] = 1
		v = LA.norm(z)*e1 - z if (z[0] < 0) else -LA.norm(z)*e1 - z
		v = v.astype(f4)
		I = np.identity(n-i)
		P = I - 2*v.dot(v.T) / v.T.dot(v)
		Q = block_diag(np.identity(i), P).astype(f4)  #put P in bottom right corner of identity
		R = Q.dot(R)
		if i == 0:
			Q_acc = Q
		else:
			Q_acc[i-1:n,i-1:n] = Q[i-1:n,i-1:n].dot(Q_acc[i-1:n,i-1:n])

	#get rid of values very close to 0
	mask = np.logical_or(R > 1e-4, R < -1e-4)
	R[~mask] = 0
	return Q_acc.T, R

def back_solve(R, QTY):
	"""
	Input:
		R - an upper triangular matrix of shape (n,d)
		QTY - column vector of shape (n,1)

	Solves for B in the equation RB = Q^T*Y, essentially backsolving the equation.
	Returns B.
	"""
	if R.shape[0] != QTY.shape[0]:
		raise Exception("Error: invalid shape input")

	n,d = R.shape
	B = np.zeros(shape=(d,1))
	for r in range(d)[::-1]:
		sum_r = 0
		for c in range(r+1,d):  #add up all known R_{r,r+1}*B_{r+1} + ... R_{r,d}B_d
			sum_r += R[r,c]*B[c]
		B[r] = (QTY[r] - sum_r)/R[r,r]  #solve for B_r

	return B

def linear_regression(data):
	"""
	Credits to Yixing Lao for this function.

	Given a filename, return the weights vector B, and the RSME as a tuple
	"""
	# with open(filename, "rb") as csvfile:
	# 	reader = csv.reader(csvfile , delimiter=",")
	# 	data = np.array([[float(r) for r in row] for row in reader])
	# data = utils.get_matrix(filename, converters)

	# create random sample train (60%) and test set (40%)
	np.random.seed(0)
	np.random.shuffle(data)
	train_num = int(data.shape[0] * 0.6)
	X_train = data[:train_num, :-1]
	Y_train = data[:train_num, -1]
	X_test = data[train_num:, :-1]
	Y_test = data[train_num:, -1]

	# linear least square Y = X beta
	Q, R = qr_decompose(X_train)
	beta = back_solve(R, np.dot(Q.T, Y_train))
	# print LA.pinv(R).dot(np.dot(Q.T, Y_train))  #actual backsolve answer

	# root mean square error
	Y_predict = np.dot(X_test, beta).reshape((Y_test.shape[0],))
	rsme = np.sqrt(np.mean((Y_predict - Y_test) ** 2))
	return beta, rsme

#########

def convertAbalone(e):
	if e=="M":
		return 0
	elif e=="F":
		return 1
	elif e=="I":
		return 2

ac = {0: convertAbalone}
abalone = utils.get_matrix("abalone.csv", converters=ac)

#get masks for M, F, and I individually
genders = abalone[:,0]
M = (genders == 0).astype(int)
F = (genders == 1).astype(int)
I = (genders == 2).astype(int)

#apply the masks to get individual cols for M, F, I
abalone[:,0] = M
abalone = np.insert(abalone, 0, F, axis=1)
abalone = np.insert(abalone, 0, I, axis=1)

files = ["abalone.csv", "regression-0.05.csv", "regression-A.csv",
"regression-B.csv", "regression-C.csv"]
datas = [abalone]
datas2 = [utils.get_matrix(f) for f in files[1:]]
datas.extend(datas2)
i = 0
for d in datas:
	b, rsme = linear_regression(d)
	print files[i], rsme
	i += 1
