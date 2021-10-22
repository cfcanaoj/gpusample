#!/bin/bash
#PBS -q dgx-full
#PBS -N job_sample2
#PBS -l walltime=00:00:60
#PBS -l select=1:ngpus=1:ncpus=1
module load cuda-toolkit
cd $PBS_O_WORKDIR
./sample3.gpux > sample3.gpulog

./sample3.cpux > sample3.cpulog
