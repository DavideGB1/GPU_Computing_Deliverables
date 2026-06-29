#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --job-name=SpMV
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [ -z "$1" ]; then
    echo "Use: sbatch run.sh <path/matrix.mtx>"
    exit 1
fi

make "$1"
