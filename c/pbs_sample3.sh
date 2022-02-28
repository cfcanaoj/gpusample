#! /bin/bash
#SBATCH --partition=dgx-full
#SBATCH --gres=gpu:1
./sample3.gpux > sample3.gpulog
./sample3.cpux > sample3.cpulog
