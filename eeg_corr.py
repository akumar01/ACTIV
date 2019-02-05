import sys, os
import argparse
import numpy as np
import h5py
import time
import pdb
import json

from utils import count_leading_trailing_true
from scipy.stats import pearsonr

total_start = time.time()
 
### parse arguments ###
parser = argparse.ArgumentParser()

parser.add_argument('data_path', default=None)
parser.add_argument('jobdir', default=None)
parser.add_argument('jobtype', default=None)
parser.add_argument('jobindex', default=None, type = int)

args = parser.parse_args()

jobdir = args.jobdir
jobtype = args.jobtype
jobindex = args.jobindex

# Initialize results file:
results_file = '%s/%s%d.h5' % (jobdir, jobtype, jobindex)
results = h5py.File(results_file, 'w')

# Load the needed portion of the data file

# NOTE: By default, just grabbing the data of the first patient located
# in the data file

# Using swmr functionality to allow multiple jobs to access the same file simultaneously
with h5py.File(args.data_path, 'r', libver = 'latest', swmr = True) as f:
	if 'channels' in jobtype:
		ref = f['cfg_PAINT_cond']['ChanERSP'][0][0]

	elif 'sources' in jobtype:
		ref = f['cfg_PAINT_cond']['SourceERSP'][0][0]

	data = f[ref][:]	
	ERSPtime = f['cfg_PAINT_cond']['ERSPtime'][:]

# Different time series have different paddings of nan values. Keep track of the usable times
valid_times = []

# ERSP shapes are (time, freqs, channels/sources)
if jobtype == 'channels' or jobtype == 'sources':

	# An individual job only calaculates one row of the correlation block
	corr_matrix = np.zeros((data.shape[2], data.shape[1], data.shape[1]))

	# For each source node
	for i in range(corr_matrix.shape[0]):
		valid_times.append([])
		# For each pairs of frequency
		start_time = time.time()
		for j in range(corr_matrix.shape[1]):
			valid_times[i].append([])
			for k in range(corr_matrix.shape[2]):
				valid_times[i][j].append([])
				# Strip nans:
				x = data[:, j, jobindex]
				y = data[:, k, i]
				nan_x = np.isnan(x)
				nan_y = np.isnan(y)

				nan_mask = np.logical_or(nan_x, nan_y)

				# Keep track of number of leading and trailing nans
				(leading_nans, trailing_nans) = \
				count_leading_trailing_true(nan_mask)

				x = x[~nan_mask]
				y = y[~nan_mask]

				valid_times[i][j][k].append(leading_nans)
				valid_times[i][j][k].append(trailing_nans)

				r2, p = pearsonr(x, y)
				corr_matrix[i, j, k] = r2
		print(time.time() - start_time)

else:

	corr_matrix = np.zeros((data.shape[1], data.shape[2], data.shape[2]))

	# For each frequency
	for i in range(corr_matrix.shape[0]):
		# For each source/channel node
		start_time = time.time()
		for j in range(corr_matrix.shape[1]):
			for k in range(corr_matrix.shape[2]):
				r2, p = pearsonr(data[:, jobindex, j], data[:, i, k])
				if np.isnan(r2):
					r2 = 1
				corr_matrix[i, j, k] = r2
		print(time.time() - start_time)

results['corr_block'] = corr_matrix
results['valid_times'] = valid_times
results.close()
print('Total runtime: %f' % (time.time() - total_start))


