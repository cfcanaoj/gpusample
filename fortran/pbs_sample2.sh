#! /bin/bash
#SBATCH --partition=dgx-full
#SBATCH --gres=gpu:1
module load nvhpc
./sample2.gpux > sample2.gpulog
./sample2.cpux > sample2.cpulog
