import numpy as np
import h5py

from pyuoi.linear_model import UoILasso


# Data should be in shape (samples, features)
# symm: whether to return the symmetric part of obtained
# correlations

def form_graph(data, symm  = True):

	uoi = UoI_Lasso(
		normalize=True,
		n_boots_sel=48,
		n_boots_est=48,
		estimation_score='r2',
		stability_selection = 1
		)

	# Fit the response of each node as a linear combination of all others. 
	n = data.shape[0]
	p = data.shape[1]

	# Note pxp shape because we will also fit an intercept (check if this is the first element
	# in the array)
	model_coeffs = np.zeros((p, p))

	for i in range(data.shape[-1]):
		X = data[:, np.arange(p) != i]
		uoi.fit(X, data[:, i])
		model_coeffs[i, :] = uoi.coef_

	# Symmetrize 
	model_coeffs = 1/2 * (model_coeffs + model_coeffs.T)
	return model_coeffs



if __name__ == '__main__':
	# Given a window size and pre_stim_window_end and
	# post_stim_window_start, construct equal time linear
	# models over channel TEP and channel ERSP recordings,
	# respectively. 

	window_size = []
	pre_stim_window_end = []
	post_stim_window_start = []
	data_file = ''

	f = h5py.File('data_file', 'r')

	# Keep as 32 bit floating point integers
	ERSP_refs = f['cfg_PAINT_cond']['ChanERSP'][:]
	TEP_refs = f['cfg_PAINT_cond']['ChanTEP'][:]

	# Need to select windowed sections here
	#######################################
    for i in range(ERSP_refs.size):
        ERSP = f[ERSP_refs[i][0]][:]

        # Need to exclude the maximum nan padding 
        leading_nan_count = np.zeros((51, 95))
        trailing_nan_count = np.zeros((51, 95))
        for j in range(51):
            for k in range(95):
                x1, x2 = count_leading_trailing_true(np.isnan(ERSP[:, j, k]))
                leading_nan_count[j, k] = x1
                trailing_nan_count[j, k] = x2


        # Select pre and post stimulation
        leading_max = int(np.amax(leading_nan_count))
        trailing_max = int(np.amax(trailing_nan_count))

        window_size = int(window_size/samp_factor)
        pre_window_end = int((1000 - pre_shift)/samp_factor)
        post_window_start = int((1000 + post_shift)/samp_factor)

        # Ensure that we don't encroach on the nan-padding
        window_size1 = min(window_size, pre_window_end - leading_max)

        window_size2 = min(window_size, int(2500/samp_factor - trailing_max - post_window_start))

        window_size = int(min(window_size1, window_size2))

        # print(i)
        pre_stim = ERSP[pre_window_end - window_size:pre_window_end, :, :]
        post_stim = ERSP[post_window_start:post_window_start + window_size]


        # Collpase and append
        pre_stim_feature_vector = np.append(pre_stim_feature_vector, pre_stim.ravel())
        post_stim_feature_vector.append(post_stim.ravel)


	# Flatten ERSP feature vectors at a given time point
	ERSP = np.reshape(ERSP, (ERSP.shape[0], ERSP.shape[1] * ERSP.shape[2]))

	t_ERSP = f['cfg_PAINT_cond']['ERSPtime']
	t_TEP = f['cfg_PAINT_cond']['TEPtime']
