#! /bin/bash
#PBS -q dgx-full
#PBS -N job_sample1
#PBS -l walltime=00:00:60
#PBS -l select=1:ngpus=2:ncpus=8
module load cuda-toolkit
cd $PBS_O_WORKDIR
./sample1.gpux > sample1.gpulog
