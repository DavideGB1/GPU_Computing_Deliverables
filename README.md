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

## Compilation

To compile the project using CMake, run the following commands from the project root:

```bash
# Clean previous builds and configure the project
rm -rf build
cmake -B build

# Build the executable
cmake --build build
```
The compiled executable SpMV will be generated inside the build directory.

---

## Usage

Once compiled, run the executable by passing the path to your sparse matrix file as an argument:

```bash
./build/SpMV <matrix_file>
```
### Example

```bash
./build/SpMV matrices/thermomech_dM.mtx
```
## Running on a SLURM Cluster
 
If you have access to an HPC cluster managed by **SLURM**, you can submit the job using the provided batch script.
To test different matrices on the cluster, open your sbatch script and modify the argument passed to the executable in the last line:
```bash
./build/SpMV matrices/your_matrix_here.mtx
```
Then, submit the job to the queue:
```bash
sbatch your_script_name.sh
```
