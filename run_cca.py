import argparse 
from cca import CCA_across_patients
import glob

data_path = '../EEGdata'

data_files = glob.glob('%s/*.mat' % data_path)

window_sizes = [200, 300]
pre_shifts = [0, 50, 100, 150, 200]
post_shifts = [0, 100, 200, 300, 400]

test_scores = np.zeros((len(window_sizes), len(pre_shifts), len(post_shifts)))
train_scores = np.zeros((len(window_sizes), len(pre_shifts), len(post_shifts)))


for i, window_size in enumerate(window_sizes):
	for j, pre_shift in enumerate(pre_shifts):
		for k, post_shift in enumerate(post_shifts):

			test_score, train_score = CCA_across_patients(data_files, alg = 'pls', 
															freq_clustering = 'single_band',
															band = 'alpha',  window_size = 500)
			test_scores[i, j, k] = test_score
			train_scores[i, j, k] = train_score
