#! /bin/bash
#PBS -q dgx-full
#PBS -N job_sample4
#PBS -l walltime=00:00:60
#PBS -l select=1:ngpus=1:ncpus=1
module load nvhpc
cd $PBS_O_WORKDIR
./sample4.gpux > sample4.gpulog
./sample4.cpux > sample4.cpulog
