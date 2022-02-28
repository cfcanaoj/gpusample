#! /bin/bash
#SBATCH --partition=dgx-full
#SBATCH --gres=gpu:1
module load nvhpc
./sample4.gpux > sample4.gpulog
./sample4.cpux > sample4.cpulog
