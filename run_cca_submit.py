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

	parser.add_argument('-jt', '--job_time', default='04:00:00')

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
	
	script_dir = '.'

	data_path = '/global/project/projectdirs/m2043/activ/stanfordEEG/DOEcollab_TMSEEGdata'

	jobdir = '/global/project/projectdirs/m2043/akumar/ACTIV/pairwise_pls'

	jobdir = '%s/%s' % (jobdir, args.jobdir)

	if not os.path.exists(jobdir):
		os.makedirs(jobdir)

	# Need this to prevent crashses
	os.system('export HDF5_USE_FILE_LOCKING=FALSE')

	# Slightly different syntax for running on cori vs. edison
	hostname = subprocess.check_output(['hostname'])

	# Hard-coded parameters
	chunk_size = 5

	# N pairs
	n_pairs = 2550

	n_jobs = int(n_pairs/chunk_size)

	for i in range(n_jobs):			

		arg_string = '%s --save_path=%s --chunk_size=%d --pair_index=%d'\
		% (data_path, jobdir, chunk_size, i)

		# Run the job interactively rather than creating all the sbatch scripts
		if args.interactive:
			if i == args.iidx:
				os.system('python %s/run_cca.py %s' % (script_dir, arg_string))
				sys.exit()
			else:
				continue
		else:
			jobname = 'pair%d' % i
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
				if 'cori'.encode() in hostname:
					sb.write('#SBATCH -C haswell')				

				sb.write('source activate nse\n')
				sb.write('srun python -u %s/run_cca.py %s' % (script_dir, arg_string))
				sb.close()
				
			# Change permissions
			os.system('chmod u+x %s' % sbname)
			if not args.test:
				if not args.first_only or i == 0:
					# Submit the job
					os.system('sbatch %s ' % sbname)




