# Sparse Matrix-Vector Multiplication (SpMV) 

This project implements **Sparse Matrix-Vector Multiplication (SpMV)** on GPU using CUDA. It reads a sparse matrix from a .mtx file, performs the multiplication against a random vector using various kernels, and outputs the results to the stats.csv file.

Owner: Davide Colosimo

---

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit

---

## Compilation

To compile the CUDA source file, run the following command from the project root:

```bash
nvcc SpMV.cu -o a.out
```

> You can replace `a.out` with any output name you prefer using the `-o` flag.

---

## Usage

Once compiled, run the executable by passing the path to your sparse matrix file as an argument:

```bash
./a.out <matrix_file>
```

### Example

```bash
./a.out my_matrix.mtx
```
## Running on a SLURM Cluster
 
If you have access to an HPC cluster managed by **SLURM**, you can submit the job using the provided batch script. To select the matrix, the user has to provide the matrix file path in the sbatch
