#!/bin/bash

#SBATCH --partition=amd-longq
#SBATCH --nodes 1
#SBATCH --mail-user=is33@hw.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1

# do some stuff to setup the environment
CUDA_VERSION=cuda10.0
CUDA_VERSION_LONG=10.0.130
CUDNN_VERSION=7.4
module purge
module load shared
module load $CUDA_VERSION/blas/$CUDA_VERSION_LONG $CUDA_VERSION/fft/$CUDA_VERSION_LONG $CUDA_VERSION/nsight/$CUDA_VERSION_LONG $CUDA_VERSION/profiler/$CUDA_VERSION_LONG $CUDA_VERSION/toolkit/$CUDA_VERSION_LONG cudnn/$CUDNN_VERSION

# execute application (read in arguments from command line)
cd /home/ishalyminov/data/dialog_knowledge_transfer &&  /home/ishalyminov/miniconda3/envs/dialog_knowledge_transfer3/bin/python -m train $@

# exit
exit 0
