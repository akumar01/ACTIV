import argparse 
from cca import CCA_across_patients
import glob
import numpy as np
import time
import pickle
import itertools
import pdb

# Command line arguments
parser = argparse.ArgumentParser()

# Window sizes
parser.add_argument('data_path', default = None)
parser.add_argument('--save_path', default = '.')
parser.add_argument('-w', '--window_sizes', nargs = '+', type = int, default = [200])
parser.add_argument('--pre_shifts', nargs = '+', type = int, default = [0, 50, 100, 150, 200])
parser.add_argument('--post_shifts', nargs = '+', type = int, default = [0, 100, 200, 300, 400]) 
#parser.add_argument('--pre_shifts', nargs = '+', type = int, default = [0])
#parser.add_argument('--post_shifts', nargs = '+', type = int, default = [0])
parser.add_argument('--chunk_size', default = 15, type = int)
parser.add_argument('--pair_index', default = 0, type = int)

args = parser.parse_args()

data_path = args.data_path
save_path = args.save_path
window_sizes = args.window_sizes
pre_shifts = args.pre_shifts
post_shifts = args.post_shifts
chunk_size = args.chunk_size
pair_index = args.pair_index

data_files = glob.glob('%s/*.mat' % data_path)

# Generate a list of tuples of all pairwise combinations of the frequency bins. Then,
# partition the list into sublists of size chunk size and specifically select the one
# corresponding to pair_index
freqs = np.arange(51)
pairs = np.array(list(itertools.permutations(freqs, 2)))

# Partition
pair_chunks = np.array_split(pairs, int(pairs.shape[0]/chunk_size))
pairs = pair_chunks[pair_index]
test_scores = np.zeros((len(window_sizes), len(pre_shifts), len(post_shifts), len(pairs)))
train_scores = np.zeros((len(window_sizes), len(pre_shifts), len(post_shifts), len(pairs)))
for i, window_size in enumerate(window_sizes):
	for j, pre_shift in enumerate(pre_shifts):
		for k, post_shift in enumerate(post_shifts):
			for l, pair in enumerate(pairs):
				start = time.time()
				test_score, train_score = CCA_across_patients(data_files, alg = 'pls', 
								freq_clustering = 'pairwise',
								pair = pair,  window_size = window_size,
								pre_shift = pre_shift, post_shift = post_shift)
				test_scores[i, j, k, l] = test_score
				train_scores[i, j, k, l] = train_score
				print('i = %d, j = %d, k = %d' % (i, j, k))
				print('%f seconds' % (time.time() - start))


filename = '%s/pls_pairwise_%d.dat' % (save_path, pair_index)
with open(filename, 'wb') as f:
	pickle.dump([test_scores, train_scores], f)
