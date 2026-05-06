#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --job-name=SpMV-thermomech_dM
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err

module load CUDA/12.5.0
module load GCC/13.3.0

rm -rf build
cmake -B build
cmake --build build

./build/SpMV matrices/thermomech_dM.mtx