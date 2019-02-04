import nump as np 
import math


# Orthogonalize the time series X and Y over time window T
# Assumes X and Y have the same length
def orth_series(X, Y, T):
	# Use a sliding window of length T, centered
	z = np.zeros(X.size - T)
	for t in range(math.ceil(T/2), X.size - math.floor(T/2)):
		parallel = X[t - math.ceil(T/2): t + math.floor(T/2)] @\
		Y[t - math.ceil(T/2): t + math.floor(T/2)].T/\
		X[t - math.ceil(T/2): t + math.floor(T/2)] @ X[t - math.ceil(T/2): t + math.floor(T/2)].T\
		* X[t]
		z[t] = Y[t] - parallel

	return z 

# Calculate the pairwise correlation between the frequency/sensor (source) pairs given by the first
# 2 axes of data. Data shape is ()
def calc_corr(data, orth = True):

