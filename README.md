# Sparse Matrix-Vector Multiplication (SpMV) 

This project implements **Sparse Matrix-Vector Multiplication (SpMV)** on GPU using CUDA. It reads a sparse matrix from a .mtx file, performs the multiplication against a random vector using various kernels, and outputs the results to the stats.csv file.

Owner: Davide Colosimo

---

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- CMake (version 3.10 or higher)
- GCC Compiler

---

## Compilation and Usage

To compile and use the project using CMake.
You need to run the following command from the project root:

```bash
make <path/matrix.mtx>
```
The compiled executable SpMV will be generated inside the build directory.

---

### Example

```bash
make matrices/thermomech_dM.mtx
```
## Running on a SLURM Cluster
 
If you have access to an HPC cluster managed by **SLURM**, you can submit the job using the provided batch script.
Then, submit the job to the queue:
```bash
sbatch run.sh <path/matrix.mtx>
```
