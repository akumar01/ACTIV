import numpy as np
import glob
import h5py
import pdb

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

def stitch_corr_matrix(datadir, corr_type='channels', average=False):

	data_files = glob.glob('%s/%s*.h5' % (datadir, corr_type))

	with h5py.File(data_files[0], 'r') as f:
		corr_blocksize = f['corr_block'].shape

	corr_matrix = np.zeros((len(data_files),) + corr_blocksize)

	for i, data_file in enumerate(data_files): 
		with h5py.File(data_file, 'r') as f:
			corr_matrix[i, :] = f['corr_block'][:]

	if average:
		N = corr_matrix.shape[-1]
		corr_matrix = 1/N**2 * np.trace(corr_matrix, axis1 = 2, axis2 = 3)
	
	return corr_matrix
		