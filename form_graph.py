print('Started execution\n')
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys, os, pdb
import time
print('Imported built ins\n')
import argparse
import subprocess

from utils import count_leading_trailing_true
print('Imported from utils\n')

from sklearn.metrics import r2_score
print('Imported sklearn\n')

hname = subprocess.check_output('hostname')

sys.path.append('/global/homes/a/akumar25')

from pyuoi.linear_model.lasso import UoI_Lasso

# Ignore Convergence Warnings
import warnings


print('Imported PyUoI')

# Data should be in shape (samples, features)
# symm: whether to return the symmetric part of obtained
# correlations

def form_graph(data, symm  = False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        uoi = UoI_Lasso(
            normalize=True,
            n_boots_sel=48,
            n_boots_est=48,
            estimation_score='r2',
            stability_selection = 1,
            fit_intercept = True
            )

        # Fit the response of each node as a linear combination of all others. 
        n = data.shape[0]
        p = data.shape[1]

        # Note pxp shape because we will also fit an intercept (check if this is the first element
        # in the array)
        model_coeffs = np.zeros((p, p - 1))

        for i in range(data.shape[-1]):
            start = time.time()
            X = data[:, np.arange(p) != i]
            print(X.shape)
            uoi.fit(X, data[:, i])
            model_coeffs[i, :] = uoi.coef_
#            Log the in-sample r2_score
            print('Time: %f' % (time.time() - start))
            print('R2 score: %f\n' % r2_score(data[:, i], uoi.predict(X)))

        # Symmetrize 
        if symm:
            model_coeffs = 1/2 * (model_coeffs + model_coeffs.T)
        return model_coeffs


def form_ERSP_graph(data_file, window_size = 500, pre_shift = 0, post_shift = 100):
    # Given a window size and pre_stim_window_end and
    # post_stim_window_start, construct equal time linear
    # models over channel TEP and channel ERSP recordings,
    # respectively. 

    f = h5py.File(data_file, 'r')

    # Keep as 32 bit floating point integers
    ERSP_refs = f['cfg_PAINT_cond']['ChanERSP'][:]

    pre_graphs = []
    post_graphs = []

    # Downsample by 10
    samp_factor = 10

    window_size = int(window_size/samp_factor)
    
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

        pre_window_end = int((1000 - pre_shift)/samp_factor)
        post_window_start = int((1000 + post_shift)/samp_factor)

        # Ensure that we don't encroach on the nan-padding
        window_size1 = min(window_size, pre_window_end - leading_max)

        window_size2 = min(window_size, int(2500/samp_factor - trailing_max - post_window_start))

        window_size_adjusted = int(min(window_size1, window_size2))
        
        # print(i)
        pre_stim = ERSP[pre_window_end - window_size_adjusted:pre_window_end, :, :]
        post_stim = ERSP[post_window_start:post_window_start + window_size_adjusted, :, :]
        
        # Switch frequency axes to be last
        pre_stim = np.swapaxes(pre_stim, 1, 2)
        post_stim = np.swapaxes(post_stim, 1, 2)
        
        # Average over cannonical frequency bands
        pre_stim_theta = np.mean(pre_stim[:, :, 0:4], axis = -1)
        pre_stim_alpha = np.mean(pre_stim[:, :, 4:8], axis = -1)
        pre_stim_beta = np.mean(pre_stim[:, :, 8:26], axis = -1)
        pre_stim_gamma = np.mean(pre_stim[:, :, 26::], axis = -1)
                                
        pre_stim = np.concatenate([pre_stim_theta, pre_stim_alpha, pre_stim_beta, pre_stim_gamma], axis = -1)

        post_stim_theta = np.mean(post_stim[:, :, 0:4], axis = -1)
        post_stim_alpha = np.mean(post_stim[:, :, 4:8], axis = -1)
        post_stim_beta = np.mean(post_stim[:, :, 8:26], axis = -1)
        post_stim_gamma = np.mean(post_stim[:, :, 26::], axis = -1)
                                
        post_stim = np.concatenate([post_stim_theta, post_stim_alpha, post_stim_beta, post_stim_gamma], axis = -1)
       
        # Collapse frequency and channels together as one set of features 
        pre_graphs.append(form_graph(pre_stim))
        post_graphs.append(form_graph(post_stim))

    return pre_graphs, post_graphs

def form_TEP_graph(data_file, window_size = 500, pre_shift = 0, post_shift = 100):
    print('Entered form TEP graph\n')
    f = h5py.File(data_file, 'r')

    # Keep as 32 bit floating point integers
    TEP_refs = f['cfg_PAINT_cond']['ChanTEP'][:]

    pre_graphs = []
    post_graphs = []

    # Downsample by 10
    samp_factor = 1

    # Need to select windowed sections here
    #######################################
    for i in range(TEP_refs.size):
        TEP = f[TEP_refs[i][0]][:]

        window_size = int(window_size/samp_factor)
        pre_window_end = int((1000 - pre_shift)/samp_factor)
        post_window_start = int((1000 + post_shift)/samp_factor)

        # print(i)
        pre_stim = TEP[pre_window_end - window_size:pre_window_end, :]
        post_stim = TEP[post_window_start:post_window_start + window_size, :]
        pre_graphs.append(form_graph(pre_stim))
        post_graphs.append(form_graph(post_stim))

    return pre_graphs, post_graphs


if __name__ == '__main__':

    # Read out command line arguments
    parser = argparse.ArgumentParser()

    # Graph from TEP or ERSP
    parser.add_argument('type')
    parser.add_argument('data_file')
    parser.add_argument('results_file')

    parser.add_argument('-w', '--window_size', default = 500)
    parser.add_argument('--pre_shift', default = 0)
    parser.add_argument('--post_shift', default = 100)

    args = parser.parse_args()

    window_size = int(args.window_size)
    pre_shift = int(args.pre_shift)
    post_shift = int(args.post_shift)

    if args.type == 'ERSP':
        pre_graphs, post_graphs = form_ERSP_graph(args.data_file, window_size, pre_shift, post_shift)
    elif args.type == 'TEP':
        pre_graphs, post_graphs = form_TEP_graph(args.data_file, window_size, pre_shift, post_shift)

    # Store results
    with h5py.File(args.results_file, 'w') as f:
        f['data_file'] = args.data_file
        f['window_size'] = window_size
        f['pre_shift'] = pre_shift
        f['post_shift'] = post_shift

        f['pre_graphs'] = pre_graphs
        f['post_graphs'] = post_graphs
