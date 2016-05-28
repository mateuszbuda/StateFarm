#!/bin/bash

# The job name is used to determine the name of job output and error files
#SBATCH -J exp2.DD2427

# Set the time allocation to be charged
#SBATCH -A edu16.SF2568

# Request a mail when the job starts and ends
#SBATCH --mail-type=ALL

# Maximum job elapsed time should be indicated whenever possible
#SBATCH -t 13:00:00

# Number of nodes that will be reserved for a given job
#SBATCH --nodes=1


#SBATCH -e error.log
#SBATCH -o output.log

#SBATCH --gres=gpu:K80:2

# Run the executable file


module load caffe/git-c6d93da
module load cuda/7.0

caffe train -solver solver.prototxt -weights VGG_16.caffemodel -gpu all
