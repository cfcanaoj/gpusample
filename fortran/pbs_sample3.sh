#! /bin/bash
#SBATCH --partition=dgx-full
#SBATCH --gres=gpu:1
module load nvhpc
./sample3.gpux > sample3.gpulog
./sample3.cpux > sample3.cpulog
