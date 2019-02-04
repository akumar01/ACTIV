import numpy as np 
from scipy.stats import pearsonr
import math
import pdb


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
# 2 axes of data. Data shape is (time, freqs, sources)
def calc_corr(data, orth = True):
	
	# Resulting correlation matrix should be across all pairs of sources and freqs (4D)
	# (sources, sources', freqs, freqs')
	corr_matrix = np.zeros((data.shape[1], data.shape[1], data.shape[2], data.shape[2]))
	
	# Iterate through each pair of (source, freq) pairs of time series, orthogonalize them,
	# and then calculate their correlations
	for i1 in range(data.shape[1]):
		for i2 in range(data.shape[1]): 
			for j1 in range(data.shape[2]):
				for j2 in range(data.shape[2]):
					if orth:
						z = orth_series(data[:, i1, j1], data[:, i2, j2], 10)
					else:
						z = data[:, i2, j2]
					r2, p = pearsonr(data[:, i1, j1], z)
					if np.isnan(r2):
						r2 = 1
					corr_matrix[i1, i2, j1, j2] = r2

	return corr_matrix