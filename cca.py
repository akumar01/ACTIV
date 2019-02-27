import sys, os
import argparse
import h5py
import time
import pdb
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split

from utils import count_leading_trailing_true

# Determine if running locally or on NERSC
import subprocess
hname = subprocess.check_output('hostname')

if ('cori'.encode() in hname) or ('edison'.encode() in hname):

    if '~/mars/signal_processing' not in sys.path:
        sys.path.append('~/mars/signal_processing')

else:
    if 'ankitnse'.encode() in hname:

        if '../mars/mars/signal_processing' not in sys.path:
            sys.path.append('../mars/mars/signal_processing')

    else:
        # And standard list of subdirectories
        if '..\\mars\\mars\\signal_processing' not in sys.path:
            sys.path.append('..\\mars\\mars\\signal_processing')

import hilbert_transform

# Calculate the Hilbert transform to use both amplitude and phase in 
# cannonical correlation analysis
def calc_hilbert_transform(file_obj):

    freqs = file_obj['cfg_PAINT_cond']['ERSPfreq'][:]
    TEPrefs = file_obj['cfg_PAINT_cond']['ChanTEP']
    TEPtrans = []
    # Assemble filter banks
    filters = []
    for freq in freqs:
        filters.append(hilbert_transform.gaussian(np.zeros((95, 2500)), 1000, freq, freq/2))
    for i in range(TEPrefs.size):
        TEP = file_obj[TEPrefs[i][0]][:]
        X = np.zeros(TEP.shape + (len(freqs),), dtype = np.complex_)
        for j in range(TEP.shape[1]):
            X[:, j, :] = hilbert_transform.hilbert_transform(TEP[:, j], 1000, filters, normalize_filters = False).T
        TEPtrans.append(X)

    return TEPtrans


# w: window size over which to calculate cannonical correlations
# post_shift: Amount by which to shift the end of the post stimulation
# window from the end. Setting this to 0 means the window is located
# as far after the stimulus as it can be

# If use_hilbert is set to true, we do our own frequency decomposition. If 
# false, then use the provided ERSPs 

# dsnum: If using our own hilbert transform, first average every #dsnum points 
# together to reduce the length of the overall time series
def cannonical_corr_analysis(data_file, window_sizes = [500], post_shifts = [0], 
                             use_hilbert = False, dsnum = 5):

    total_start = time.time()   
    corrmodels = []

    if not isinstance(window_sizes, list):
        window_sizes = [window_sizes]

    if not isinstance(post_shifts, list):
        post_shifts = [post_shifts]

    # ERSP are downsampled by a factor of 10
    if use_hilbert:
        samp_factor = dsnum
    else:
        samp_factor = 10

    with h5py.File(data_file, 'r') as f:

        # Assemble feature matrices pre and post
        # stimulation

        if use_hilbert:

            X = calc_hilbert_transform(f)
            
            # have the window extend backwards from 100 ms prior to stimulation: 
            pre_shift = 100

            for i, XX in enumerate(X):

                # Average together
                XXds = np.mean(XX.reshape((-1, dsnum, XX.shape[1], XX.shape[2])), axis = 1)
                train_scores = np.zeros((len(window_sizes), len(post_shifts)))
                test_scores = np.zeros(train_scores.shape)
                for j, w in enumerate(window_sizes):
                    for k, post_shift in enumerate(post_shifts):
                        window_size = int(w/samp_factor)
                        pre_window_end = int((1000 - pre_shift)/samp_factor)
                        post_window_start = int((1000 + post_shift)/samp_factor)

                        pre_feature_vector = np.zeros((window_size, 2, XXds.shape[1], XXds.shape[2]))
                        post_feature_vector = np.zeros(pre_feature_vector.shape)
                        
                        pre_feature_vector[:, 0, ...] = np.abs(XXds[pre_window_end - window_size:
                                                                pre_window_end,...])
                        pre_feature_vector[:, 1, ...] = np.angle(XXds[pre_window_end - window_size:
                                                                  pre_window_end,...])

                        post_feature_vector[:, 0, ...] = np.abs(XXds[post_window_start:
                                                                 post_window_start + window_size,...])
                        post_feature_vector[:, 1, ...] = np.angle(XXds[post_window_start:
                                                                   post_window_start + window_size,...])

                        # Reshape, collapsing all features together

                        pre_feature_vector = pre_feature_vector.reshape(window_size, -1)
                        post_feature_vector = post_feature_vector.reshape(window_size, -1)

                        # Split into train/test pairs to test whether CCA is overfitting
                        Xtrain, Xtest, Ytrain, Ytest = train_test_split(pre_feature_vector,
                                                                        post_feature_vector,
                                                                        test_size = 0.25)



                        # Perform cannonical correlation analysis on the basis of this data
                        corrmodel = CCA(n_components = 1)
                        corrmodel.fit(Xtrain, Ytrain)
                        train_scores[j, k] = corrmodel.score(Xtrain, Ytrain)
                        test_scores[j, k] = corrmodel.score(Xtest, Ytest)

                corrmodels.append([train_scores, test_scores])

        else:
            # ERSP time series references
            ERSP_refs = f['cfg_PAINT_cond']['ChanERSP']

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
                scores = np.zeros((len(window_sizes), len(post_shifts)))            

                for j, w in enumerate(window_sizes):
                    for k, s in enumerate(post_shifts):
                        # Window size is the the minimum of 650 ms and the time between the 
                        # first non-zero nan time and t = 0 (stimulation application)
                        window_size1 = min(w/samp_factor, 95 - leading_max)

                        window_size2 = min(w/samp_factor, 155 - s/samp_factor - trailing_max)

                        window_size = min(window_size1, window_size2)

                        # print(i)

                        pre_stim = ERSP[leading_max:leading_max + window_size, :, :]
                        post_stim = np.flip(ERSP, axis = 0)[trailing_max + 1:trailing_max + window_size + 1, :, :]

                        # Reshape, collapsing frequencies and channels
                        pre_stim = pre_stim.reshape((pre_stim.shape[0], pre_stim.shape[1] * pre_stim.shape[2]))
                        post_stim = post_stim.reshape((post_stim.shape[0], post_stim.shape[1] * post_stim.shape[2]))

                        # Perform cannonical correlation analysis on the basis of this data
                        corrmodel = CCA(n_components = 1)
                        corrmodel.fit(pre_stim, post_stim)

                        scores[j, k] = corrmodel.score(pre_stim, post_stim)
                        
            corrmodels.append(scores)

    # print("Runtime: %f seconds" % (time.time() - total_start))
    return corrmodels