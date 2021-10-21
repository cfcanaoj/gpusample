#! /bin/bash
#PBS -q dgx-full
#PBS -N job_sample1
#PBS -l walltime=00:00:60
#PBS -l select=1:ngpus=1:ncpus=1
module load cuda-toolkit/11.0
cd $PBS_O_WORKDIR
./sample2.gpux > sample2.gpulog
./sample2.cpux > sample2.cpulog
