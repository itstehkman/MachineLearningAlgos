import numpy as np
from numpy import linalg as LA
from scipy.linalg import block_diag

def qr_decompose(X):
	"""
	QR-Decompose matrix X using householder's reflection for each columnb of X.
	Returns Q,R
	"""
	n,d = X.shape
	R = X
	Q_acc = np.identity(n)

	for i in xrange(d):
		if i >= n-1:
			break
		z = R[i:n,i]
		z = np.array([z]).T  #turn 1d array into column vector
		e1 = np.zeros(shape=z.shape, dtype=int)
		e1[0] = 1
		v = LA.norm(z)*e1 - z if (z[0] < 0) else -LA.norm(z)*e1 - z
		I = np.identity(n-i)	
		P = I - 2*v.dot(v.T) / v.T.dot(v)
		Q = block_diag(np.identity(i), P)  #put P in bottom right corner of identity
		R = Q.dot(R)
		Q_acc = Q.dot(Q_acc)

	#get rid of values very close to 0
	mask = np.logical_or(R > 1e-13, R < -1e-13)
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
		for c in range(r+1,d):
			sum_r += R[r,c]*B[c]
		B[r] = (QTY[r] - sum_r)/R[r,r]

	return B

#########

X = np.array([[1, -1, -1], [1, 2, 3], [2, 1, 1], [2, -2, 1], [3, 2, 1]]) 
Y = np.array([1,2,3,4,5])
print "X:\n", X
print "Y:\n", Y
Q,R = qr_decompose(X)
print "Q:\n", Q
print "R:\n", R
print "Q.T*Y:\n", Q.T.dot(Y)

B = back_solve(R, Q.T.dot(Y))
print "B:\n", B