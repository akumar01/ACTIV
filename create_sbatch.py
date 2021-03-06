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

	# Shouldn't need more than 30 minutes
	parser.add_argument('-jt', '--job_time', default='00:30:00')

	# First only: Submit only the first job
	# Test: Create all files/folders but do not submit any of the jobs
	parser.add_argument('-f', '--first_only', action = 'store_true')
	parser.add_argument('-t', '--test', action = 'store_true')

	# Interactive execution: Execute the job given by job_index interactively
	# instead of submitting into the slurm queue
	parser.add_argument('-i', '--interactive', action = 'store_true')
	# Job index (corresponds to jobnum) to launch interactively
	parser.add_argument('--iidx', default = 0)

	args = parser.parse_args()
	
	script_dir = '/global/homes/a/akumar25/ACTIV'

	data_path = '/global/project/projectdirs/m2043/activ/stanfordEEG/DOEcollab_TMSEEGdata'

	save_path = '/global/project/projectdirs/m2043/akumar/ACTIV/pairwise_pls'

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

			# Run the job interactively rather than creating all the sbatch scripts
			if args.interactive:
				os.system('python %s/eeg_corr.py %s %s %s %d'
				% (script_dir, data_path, jobdir, jobtype, int(args.iidx)))

				sys.exit()

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



