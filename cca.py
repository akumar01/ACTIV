import sys, os
import argparse
import h5py
import time
import pdb
import numpy as np
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import r2_score

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
                             use_hilbert = False, dsnum = 5, pre_shifts = [0]):

    total_start = time.time()   

    if not isinstance(window_sizes, list):
        window_sizes = [window_sizes]

    if not isinstance(post_shifts, list):
        post_shifts = [post_shifts]

    # ERSP are downsampled by a factor of 10
    if use_hilbert:
        samp_factor = dsnum
    else:
        samp_factor = 10

    # have the window extend backwards from 100 ms prior to stimulation: 
    with h5py.File(data_file, 'r') as f:

        # Assemble feature matrices pre and post
        # stimulation

        if use_hilbert:

            X = calc_hilbert_transform(f)
            
            corrmodels_all = []
            corrmodels_amp = []
            corrmodels_phase = []

            for i, XX in enumerate(X):

                # Average together
                XXds = np.mean(XX.reshape((-1, dsnum, XX.shape[1], XX.shape[2])), axis = 1)
                test_scores = np.zeros((len(window_sizes), len(post_shifts), len(pre_shifts)))
                train_scores = np.zeros(test_scores.shape)

                test_scores_amp = np.zeros(test_scores.shape)
                train_scores_amp = np.zeros(test_scores.shape)

                test_scores_phase = np.zeros(test_scores.shape)
                train_scores_phase = np.zeros(train_scores.shape)

                for j, w in enumerate(window_sizes):
                    for k, post_shift in enumerate(post_shifts):
                        for l, pre_shift in enumerate(pre_shifts):
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
                            pre_all_features = pre_feature_vector.reshape(window_size, -1)
                            post_all_features = post_feature_vector.reshape(window_size, -1)

                            # Perform a cross-validated cannonical correlation analysis on the basis of this data
                            corrmodel = CCA(n_components = 1)

                            crsval = cross_validate(corrmodel, pre_all_features, post_all_features, cv = 5,
                                                    return_train_score = True)

                            test_scores[j, k, l] = np.mean(crsval['test_scores'])
                            train_scores[j, k, l] = np.mean(crsval['train_scores'])


                            # Consider only the amplitudes
                            pre_amp_features = pre_feature_vector[:, 0, :, :].reshape(window_size, -1)
                            post_amp_features = post_feature_vector[:, 0, :, :].reshape(window_size, -1)

                            crsval = cross_validate(corrmodel, pre_amp_features, post_amp_features, cv = 5,
                                                    return_train_score = True)

                            test_scores_amp[j, k, l] = np.mean(crsval['test_scores'])
                            train_scores_amp[j, k, l] = np.mean(crsval['train_scores'])

                            # COnsider only the phases
                            pre_phase_features = pre_feature_vector[:, 1, :, :].reshape(window_size, -1)
                            post_phase_features = post_feature_vector[:, 1, :, :].reshape(window_size, -1)

                            crsval = cross_validate(corrmodel, pre_phase_features, post_phase_features, cv = 5,
                                                    return_train_score = True)

                            test_scores_phase[j, k, l] = np.mean(crsval['test_scores'])
                            train_scores_phase[j, k, l] = np.mean(crsval['train_scores'])


                corrmodels_all.append([test_scores, train_scores])
                corrmodels_amp.append([test_scores, train_scores])
                corrmodels_phase.append([test_scores, train_scores])

            return corrmodels_all, corrmodels_amp, corrmodels_phase

        else:

            corrmodels = []

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
                test_scores = np.zeros((len(window_sizes), len(post_shifts), len(pre_shifts)))
                train_scores = np.zeros(test_scores.shape)
                for j, w in enumerate(window_sizes):
                    for k, s in enumerate(post_shifts):
                        for l, pre_shift in enumerate(pre_shifts):
                            window_size = int(w/samp_factor)
                            pre_window_end = int((1000 - pre_shift)/samp_factor)
                            post_window_start = int((1000 + s)/samp_factor)

                            # Ensure that we don't encroach on the nan-padding
                            window_size1 = min(window_size, pre_window_end - leading_max)

                            window_size2 = min(window_size, int(2500/samp_factor - trailing_max - post_window_start))

                            window_size = int(min(window_size1, window_size2))

                            # print(i)
                            pre_stim = ERSP[pre_window_end - window_size:pre_window_end, :, :]
                            post_stim = ERSP[post_window_start:post_window_start + window_size]

                            # Reshape, collapsing frequencies and channels
                            pre_stim = pre_stim.reshape((pre_stim.shape[0], pre_stim.shape[1] * pre_stim.shape[2]))
                            post_stim = post_stim.reshape((post_stim.shape[0], post_stim.shape[1] * post_stim.shape[2]))
                            
                            # Perform a cross-validated cannonical correlation analysis on the basis of this data
                            corrmodel = CCA(n_components = 1)
                            crsval = cross_validate(corrmodel, pre_feature_vector, post_feature_vector, cv = 5,
                                                    return_train_score = True)

                            test_scores[j, k, l] = np.mean(crsval['test_scores'])
                            train_scores[j, k, l] = np.mean(crsval['train_scores'])

                corrmodels.append(scores)



            return corrmodels

# Do cross decomposition across trials. Select algorithm to be either CCA or PLSRegression
# Freq clustering: Need to reduce the number of features by grouping frequencies together
# 'cannonical' groups by cannonical neuroscience bands
# 'equal' groups by equally sized bins that are specified using bin_size
# 'random' groups by random selections of bins of size given by bin_size
# 'single_band': See whether individual cannonical bands predict themselves pre and post
#  specify the band using the band argument
# 'pairwise': Select individual pairs of bands pre and post stimulation. Specify the pair as a tuple
# using the pair argument
def CCA_across_patients(data_files, alg = 'cca', freq_clustering = 'cannonical', bin_size = 10,
                        window_size = 500, post_shift = 0, pre_shift = 0, band = 'alpha', pair = (1, 1)):

    # Assemble the set of feature vectors

    # Send the arguments in units of ms
    samp_factor = 10
    window_size = int(window_size/samp_factor)
    pre_shift = pre_shift/samp_factor
    post_shift = post_shift/samp_factor
    
    pre_stim_feature_vector = np.array([])
    post_stim_feature_vector = np.array([])

    for data_file in data_files:

        with h5py.File(data_file, 'r') as f:

            # ERSP time series references
            ERSP_refs = f['cfg_PAINT_cond']['ChanERSP']

            for i in range(ERSP_refs.size):
                # Use 32 bit floating precision
                ERSP = np.zeros((250, 51, 95), dtype = np.float64)
                f[ERSP_refs[i][0]].read_direct(ERSP)

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

                pre_window_end = int(1000/samp_factor - pre_shift)
                post_window_start = int(1000/samp_factor + post_shift)

                # Ensure that we don't encroach on the nan-padding
                window_size1 = min(window_size, pre_window_end - leading_max)

                window_size2 = min(window_size, int(2500/samp_factor - trailing_max - post_window_start))

                window_size = int(min(window_size1, window_size2))

                pre_stim = ERSP[pre_window_end - window_size:pre_window_end, :, :]
                post_stim = ERSP[post_window_start: post_window_start + window_size, :, :]
                
                # Re-arrange axes so that frequency bins are last
                pre_stim = np.swapaxes(pre_stim, 1, 2)
                post_stim = np.swapaxes(post_stim, 1, 2)

                if freq_clustering == 'cannonical':

                    # Average across cannonical frequency bands
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
                elif freq_clustering == 'equal':
                    # Chop off the lowest frequency bin so we have a non-prime number of bins...
                    pre_stim = pre_stim[..., 1::]
                    post_stim = post_stim[..., 1::]

                    # Average across equal number of frequency bands
                    pre_stim = np.mean(pre_stim.reshape((pre_stim.shape[0], pre_stim.shape[1], -1, bin_size)), axis = -1)                              
                    post_stim = np.mean(post_stim.reshape((post_stim.shape[0], post_stim.shape[1], -1, bin_size)), axis = -1)

                    # Collapse
                    pre_stim = pre_stim.reshape((pre_stim.shape[0], pre_stim.shape[1] * pre_stim.shape[2]))
                    post_stim = post_stim.reshape((post_stim.shape[0], post_stim.shape[1] * post_stim.shape[2]))
                    
                elif freq_clustering == 'random':
                    
                    # Chop off the lowest frequency bin so we have a non-prime number of bins...
                    pre_stim = pre_stim[..., 1::]
                    post_stim = post_stim[..., 1::]
                    
                    # Average across random collection of frequency bins
                    idxs = np.arange(pre_stim.shape[-1])
                    np.random.shuffle(idxs)
                    idxs = np.split(idxs, int(pre_stim.shape[-1]/bin_size))

                    pre_stim_rand1 = np.mean(pre_stim[:, :, idxs[0]], axis = -1)
                    pre_stim_rand2 = np.mean(pre_stim[:, :, idxs[1]], axis = -1)
                    pre_stim_rand3 = np.mean(pre_stim[:, :, idxs[2]], axis = -1)
                    pre_stim_rand4 = np.mean(pre_stim[:, :, idxs[3]], axis = -1)
                    pre_stim_rand5 = np.mean(pre_stim[:, :, idxs[4]], axis = -1)

                    pre_stim = np.concatenate([pre_stim_rand1, pre_stim_rand2, pre_stim_rand3, pre_stim_rand4, pre_stim_rand5], axis = -1)
                
                    post_stim_rand1 = np.mean(post_stim[:, :, idxs[0]], axis = -1)
                    post_stim_rand2 = np.mean(post_stim[:, :, idxs[1]], axis = -1)
                    post_stim_rand3 = np.mean(post_stim[:, :, idxs[2]], axis = -1)
                    post_stim_rand4 = np.mean(post_stim[:, :, idxs[3]], axis = -1)
                    post_stim_rand5 = np.mean(post_stim[:, :, idxs[4]], axis = -1)

                    post_stim = np.concatenate([post_stim_rand1, post_stim_rand2, post_stim_rand3, post_stim_rand4, post_stim_rand5], axis = -1)
                elif freq_clustering == 'single_band':

                    if band == 'theta':
                        pre_stim = pre_stim[:, :, 0:4]
                        post_stim = post_stim[:, :, 0:4]
                    elif band == 'alpha':
                        pre_stim = pre_stim[:, :, 4:8]
                        post_stim = post_stim[:, :, 4:8]
                    elif band == 'beta':
                        pre_stim = pre_stim[:, :, 8:26]
                        post_stim = post_stim[:, :, 8:26]
                    elif band == 'gamma':
                        pre_stim = pre_stim[:, :, 26::]
                        post_stim = post_stim[:, :, 26::]
                    elif band == 'all':
                        pass
                elif freq_clustering == 'pairwise':

                    pre_stim = pre_stim[:, :, pair[0]]
                    post_stim = post_stim[:, :, pair[1]]

                # Collpase and append
                if pre_stim_feature_vector.size == 0:
                    pre_stim_feature_vector = np.append(pre_stim_feature_vector, pre_stim.reshape((1, -1)))
                    post_stim_feature_vector = np.append(post_stim_feature_vector, post_stim.reshape((1, -1)))
                                                
                    pre_stim_feature_vector = pre_stim_feature_vector.reshape((1, -1))
                    post_stim_feature_vector = post_stim_feature_vector.reshape((1, -1))
                else: 
                    pre_stim_feature_vector = np.concatenate([pre_stim_feature_vector, pre_stim.reshape((1, -1))])
                    post_stim_feature_vector = np.concatenate([post_stim_feature_vector, post_stim.reshape((1, -1))])
                    

    # Convert to 32 bit floating precision
    pre_stim_feature_vector = pre_stim_feature_vector.astype(np.float32)
    post_stim_feature_vector = post_stim_feature_vector.astype(np.float32)

    # Attempt to do a cross-validated CCA across all the features
    # Perform a cross-validated cannonical correlation analysis on the basis of this data

    if alg == 'cca':
        corrmodel = CCA(n_components = 1)
        crsval = cross_validate(corrmodel, pre_stim_feature_vector, post_stim_feature_vector, cv = 5,
                return_train_score = True)
        return np.mean(crsval['test_score']), np.mean(crsval['train_score'])

    elif alg == 'pls':
        corrmodel = PLSRegression()
        # Manually cross-validate
        folds = KFold(n_splits = 5)
        test_scores = []
        train_scores = []
        for train_index, test_index in folds.split(pre_stim_feature_vector, post_stim_feature_vector):
            corrmodel.fit(pre_stim_feature_vector[train_index], post_stim_feature_vector[train_index])
            test_scores.append(corrmodel.score(pre_stim_feature_vector[test_index], post_stim_feature_vector[test_index]))
            train_scores.append(corrmodel.score(pre_stim_feature_vector[train_index], post_stim_feature_vector[train_index]))                
        return np.mean(test_scores), np.mean(train_scores)