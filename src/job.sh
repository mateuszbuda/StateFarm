#!/bin/bash

# The job name is used to determine the name of job output and error files
#SBATCH -J DD2427

# Set the time allocation to be charged
#SBATCH -A allocation

# Request a mail when the job starts and ends
#SBATCH --mail-type=ALL

# Maximum job elapsed time should be indicated whenever possible
#SBATCH -t hh:mm:ss

# Number of nodes that will be reserved for a given job
#SBATCH --nodes=1


#SBATCH -e error.log
#SBATCH -o output.o

#SBATCH --gres=gpu:K80:2

# Run the executable file

module load caffe/git-c6d93da
module load cuda/7.0

python extract_features.py --model_def VGG_16_deploy.prototxt --model VGG_16.caffemodel
