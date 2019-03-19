import argparse 
from cca import CCA_across_patients
import glob
import numpy as np
import time
import pickle

data_path = '../EEGdata'

data_files = glob.glob('%s/*.mat' % data_path)

window_sizes = [200]
pre_shifts = [0, 50, 100, 150, 200]
post_shifts = [0, 100, 200, 300, 400]

test_scores = np.zeros((len(window_sizes), len(pre_shifts), len(post_shifts)))
train_scores = np.zeros((len(window_sizes), len(pre_shifts), len(post_shifts)))

for i, window_size in enumerate(window_sizes):
	for j, pre_shift in enumerate(pre_shifts):
		for k, post_shift in enumerate(post_shifts):

			start = time.time()
			test_score, train_score = CCA_across_patients(data_files, alg = 'pls', 
							freq_clustering = 'pairwise',
							band = (1, 2),  window_size = window_size,
							pre_shift = pre_shift, post_shift = post_shift)
			test_scores[i, j, k] = test_score
			train_scores[i, j, k] = train_score
			print('i = %d, j = %d, k = %d' % (i, j, k))
			print('%f seconds' % (time.time() - start))


filename = 'pls_pairwise.dat'
with open(filename, 'wb') as f:
	pickle.dump([test_scores, train_scores], f)
