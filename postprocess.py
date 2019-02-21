import numpy as np
import glob
import h5py
import pdb
import re

# Stitch together the correlation matrix from separate blocks

# Valid corr_types

# channels: Pairwise correlations between channels, setting average
# flag to True will average over frequencies

# freq-channels: Pairwise correlations between frequncies, 
# setting average flag to True will average over channels

# freq-sources: Pairwise correlations between frequencies, 
# setting average flag to True will average over sources

# sources: Pairwise correlations between sources, setting average
# flag to True will average over all frequencies


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    # Strip .h5 extension
    s = s[:-3]
    return int(re.split('([0-9]+)', s)[-2])

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l

def stitch_corr_matrix(datadir, corr_type='channels', average=False):

	data_files = glob.glob('%s/%s*.h5' % (datadir, corr_type))

	data_files = sort_nicely(data_files)

	with h5py.File(data_files[0], 'r') as f:
		corr_blocksize = f['corr_block'].shape

	corr_matrix = np.zeros((len(data_files),) + corr_blocksize)

	for i, data_file in enumerate(data_files): 
		with h5py.File(data_file, 'r') as f:
			corr_matrix[i, :] = f['corr_block'][:]
			if np.sum(np.diag(corr_matrix[i, i, :, :, 0])) != 51:
				pdb.set_trace()

	if average:
		N = corr_matrix.shape[-1]
		corr_matrix = 1/N**2 * np.trace(corr_matrix, axis1 = 2, axis2 = 3)
	
	return corr_matrix
		