import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys, os, pdb
import time
from utils import count_leading_trailing_true

# Hack to import pyuoi
parent_path, current_dir = os.path.split(os.path.abspath('.'))
while current_dir not in ['nse']:
    parent_path, current_dir = os.path.split(parent_path)
p = os.path.join(parent_path, current_dir)
# Add analysis
if p not in sys.path:
    sys.path.append(p)

import subprocess
hname = subprocess.check_output('hostname')


if 'ankitnse'.encode() in hname:

    if '%s/uoicorr' % p not in sys.path:
        sys.path.append('%s/uoicorr' % p)
    if '%s/PyUoI' % p not in sys.path:
        sys.path.append('%s/PyUoI' % p)

else:
    # And standard list of subdirectories
    if '%s\\pyuoi' % p not in sys.path:
        sys.path.append('%s\\pyuoi' % p)
    if '%s\\uoicorr' % p not in sys.path:
        sys.path.append('%s\\uoicorr' % p)

from pyuoi.linear_model.lasso import UoI_Lasso

# Data should be in shape (samples, features)
# symm: whether to return the symmetric part of obtained
# correlations

def form_graph(data, symm  = True):
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
        uoi.fit(X, data[:, i])
        model_coeffs[i, :] = uoi.coef_
        print(time.time() - start)
    # Symmetrize 
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
        post_stim = ERSP[post_window_start:post_window_start + window_size, :, :]
        # Collapse frequency and channels together as one set of features 
        pre_stim = pre_stim.reshape((pre_stim.shape[0], pre_stim.shape[1] * pre_stim.shape[2]))
        post_stim = post_stim.reshape((post_stim.shape[0], post_stim.shape[1] * post_stim.shape[2]))

        pre_graphs.append(form_graph(pre_stim))
        post_graphs.append(form_graph(post_stim))

    return pre_graphs, post_graphs

def form_TEP_graph(data_file, window_size = 500, pre_shift = 0, post_shift = 100):

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
