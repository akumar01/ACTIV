import sys, os, errno
from datetime import datetime
import subprocess
import shlex
import pdb
import itertools
import glob
import argparse
import json
import importlib
import subprocess
import h5py


if __name__ == '__main__':
	# Create and execute an sbatch script correlation calculations

	parser = argparse.ArgumentParser()

	# Name of folder to put the results in (by default, working in
	# /global/project/projectdirs/m2043/ankit/ACTIV/correlations)

	parser.add_argument('jobdir', default = None)

	# Name of data file to process
	parser.add_argument('-d', '--datfile', default = None)

	# c (channels): Calculate pairwise correlations between channels, averaged 
	# over all frequencies

	# fc (freq-channels): Calculate pairwise correlations between frequncies, 
	# averaged over all channels

	# fs (freq-sources): Calculate pairwise correlations between frequencies, 
	# averaged over all sources

	# s (sources): Calculate pairwise correlations between sources, averaged
	# over all frequencies

	# Level of parallelization: Each distinct job will calculate a single row
	# of the corresponding correlation matrix

	parser.add_argument('-c', '--channels', action = 'store_true')
	parser.add_argument('-fc', '--freq-channels', action = 'store_true')
	parser.add_argument('-fs', '--freq-sources', action = 'store_true')
	parser.add_argument('-s', '--sources', action = 'store_true')

	# Shouldn't need more than 30 minutes
	parser.add_argument('-jt', '--job_time', default='00:30:00')

	# First only: Submit only the first job
	# Test: Create all files/folders but do not submit any of the jobs
	parser.add_argument('-f', '--first_only', action = 'store_true')
	parser.add_argument('-t', '--test', action = 'store_true')


	args = parser.parse_args()
	
	script_dir = '/global/homes/a/akumar25/ACTIV'

	data_path = '/global/project/projectdirs/m2043/activ/stanfordEEG/DOEcollab_TMSEEGdata'

	# Grab the first data file in the directory for testing purposes:
	if args.datfile is None:
		args.datfile = os.listdir(data_path)[0]

	data_path = '%s/%s' % (data_path, args.datfile)

	# ensure that data_file exists:
	if not os.path.isfile(data_path):
		raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.datfile)

	# Get number of channels, sources, and frequencies:
	with h5py.File(data_path, 'r') as f:
		nchans = int(f['cfg_PAINT_cond']['channum'][0][0])
		nsources = int(f['cfg_PAINT_cond']['sourcenum'][0][0])
		nfreqs = int(f['cfg_PAINT_cond']['ERSPfreq'].size)

	# Number of rows of the correlation matrix for the different job types
	jobnums = [nchans, nfreqs, nfreqs, nchans]

	jobdir = '/global/project/projectdirs/m2043/akumar/ACTIV/correlations'

	jobdir = '%s/%s' % (jobdir, args.jobdir)

	if not os.path.exists(jobdir):
		os.makedirs(jobdir)

	# Log all details
	log_file = open('%s/log.txt' % jobdir, 'w')
	log_file.write('Jobs submitted at ' + "{:%H:%M:%S, %B %d, %Y}".format(datetime.now()) + '\n\n\n')

	log_file.write('Data file Processed: %s\n' % args.datfile)

	# Need this to prevent crashses
	os.system('export HDF5_USE_FILE_LOCKING=FALSE')

	hostname = subprocess.check_output(['hostname'])

	for i, jobtype in enumerate(['channels', 'freq_channels', 'freq_sources', 'sources']):

		if getattr(args, jobtype):

			for i in range(jobnums[i]):
				jobname = '%s%d' % (jobtype, i)
				sbname = '%s/sbatch_%s.sh' % (jobdir, jobname)
				with open(sbname, 'w') as sb:
					# Arguments common across jobs
					sb.write('#!/bin/bash\n')
					sb.write('#SBATCH -q shared\n')
					sb.write('#SBATCH -n 1\n')
					sb.write('#SBATCH -t %s\n' % args.job_time)

					sb.write('#SBATCH --job-name=%s\n' % jobname)
					sb.write('#SBATCH --out=%s/%s.o\n' % (jobdir, jobname))
					sb.write('#SBATCH --error=%s/%s.e\n' % (jobdir, jobname))
					sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
					sb.write('#SBATCH --mail-type=FAIL\n')
					# Load python and any other necessary modules
					sb.write('module load python/3.6-anaconda-4.4\n')
					# script(s) to actually run
					if 'cori'.encode() in hostname:
						sb.write('srun -C haswell python3  %s/eeg_corr.py %s %s %s %d' 
							% (script_dir, data_path, jobdir, jobtype, i))
					else:
						sb.write('srun python3  %s/eeg_corr.py %s %s %s %d' 
							% (script_dir, data_path, jobdir, jobtype, i))
					sb.close()
					
				# Change permissions
				os.system('chmod u+x %s' % sbname)
				if not args.test:
					if not args.first_only or i == 0:
						# Submit the job
						os.system('sbatch %s ' % sbname)

	log_file.close()



