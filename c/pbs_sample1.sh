#! /bin/bash
#SBATCH --partition=dgx-full
#SBATCH --gres=gpu:1
./sample1.gpux > sample1.gpulog
./sample1.cpux > sample1.cpulog
