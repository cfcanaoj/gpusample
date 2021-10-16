#! /bin/bash
#PBS -q dgx-full
#PBS -N job_sample1
#PBS -l walltime=00:00:60
#PBS -l select=1:ngpus=1:ncpus=2
module load nvhpc
cd $PBS_O_WORKDIR
mpiexec -n 2 sample1.gpux
