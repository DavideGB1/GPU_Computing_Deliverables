#include <cusparse.h>
#include "matrix_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "time/time_lib.h"
#include <dirent.h>

#define WARMUP 2
#define NITER 10
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256

__global__ void SpMV_COO(
    int nnz,
    const int    *__restrict__ row,
    const int    *__restrict__ col,
    const double *__restrict__ val,
    const double *__restrict__ vector,
    double *res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <nnz) {
        double partial_prod = val[i] * vector[col[i]];
        atomicAdd(&res[row[i]], partial_prod);
    }
}

__global__ void SpMV_CSR_Scalar(
    int n_row,
    const int    *__restrict__ row_ptr,
    const int    *__restrict__ col_ind,
    const double *__restrict__ values,
    const double *__restrict__ vector,
    double *res)
{
    //Use grid of size n_row*1
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < n_row) {
        double sum = 0;
        for(int j = row_ptr[thread_id]; j<row_ptr[thread_id+1]; j++) {
            sum += values[j]*vector[col_ind[j]];
        }
        res[thread_id] = sum;
    }
}

__global__ void SpMV_CSR_Vector(
    int n_row,
    const int    *__restrict__ row_ptr,
    const int    *__restrict__ col_ind,
    const double *__restrict__ values,
    const double *__restrict__ vector,
    double *res)
{
    //Use grid of size tbd
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = thread_id/32;
    int lane_id = thread_id%32;
    thread_id  = threadIdx.x;
    __shared__ double sum[THREADS_PER_BLOCK];
    sum[thread_id] = 0;
    if (warp_id < n_row){
        for(int j = row_ptr[warp_id]+lane_id; j<row_ptr[warp_id+1]; j+=WARP_SIZE) {
            sum[thread_id] += values[j]*vector[col_ind[j]];
        }
    }
     __syncthreads();
    if (lane_id <16){ sum[thread_id] += sum[thread_id+16]; }
    __syncwarp();
    if (lane_id <8){ sum[thread_id] += sum[thread_id+8]; }
    __syncwarp();
    if (lane_id <4){ sum[thread_id] += sum[thread_id+4]; }
    __syncwarp();
    if (lane_id <2){ sum[thread_id] += sum[thread_id+2]; }
    __syncwarp();
    if (lane_id ==0){
        sum[thread_id] += sum[thread_id+1];
        res[warp_id] = sum[thread_id];
    }
}

void my_SpMV(CSR_Matrix *csr, double *vector, double *res) {
#   pragma omp parallel for
    for (int i = 0; i < csr->n_row; i++) {
        double sum = 0;
        for (int j = csr->row_ptr[i]; j < csr->row_ptr[i+1]; j++) {
            sum += csr->values[j] * vector[csr->col_ind[j]];
        }
        res[i] = sum;
    }
}
void print_stats(const char *label, double *timers) {

}

//COO
//COO-Sorted
//CSR-Scalar
//CSR-Vector
int main(int argc, char *argv[]) {
    if (argc != 2) {
		fprintf(stderr, "Use: %s matrix_file.mtx\n", argv[0]);
		return 1;
	}

    TIMER_DEF(0);
	long seed = time(NULL);
    srand(seed);

    FILE *f = fopen(argv[1], "r");
    if (f == NULL) {
		perror("Error: file not found\n");
		return 1;
	}

    TIMER_START(0);
    COO_Matrix *coo = mm_parser(f);
    TIMER_STOP(0);
    printf("Matrix reading time: %lfs\n", TIMER_ELAPSED(0)/1.e6);
    fclose(f);

    TIMER_START(0);
    CSR_Matrix *csr = create_CSR(coo);
    TIMER_STOP(0);
    printf("CSR Creation Time: %lfs\n", TIMER_ELAPSED(0)/1.e6);

    TIMER_START(0);
    double *vector = (double *)malloc(csr->n_col * sizeof(double));
    for (int i = 0; i < csr->n_col; i++){
		vector[i] = (double)rand() / 1.e3;
	}
    TIMER_STOP(0);
    printf("Vector Generation Time: %lfs\n", TIMER_ELAPSED(0)/1.e6);

    double *res = (double *)calloc(csr->n_row, sizeof(double));
    double timers[NITER];

    printf("Matrix: %s  rows=%d  cols=%d  nnz=%d\n\n", argv[1], csr->n_row, csr->n_col, csr->nnz);
    //CPU
    printf("CPU SpMV\n");
    for (int i=-WARMUP; i<NITER; i++) {
        TIMER_START(0);
        my_SpMV(csr,vector,res);
        TIMER_STOP(0);

        double iter_time = TIMER_ELAPSED(0) / 1.e6;
        if( i >= 0) timers[i] = iter_time;

        printf("Iteration %d tooks %lfs\n", i, iter_time);
    }
    printf("\n");
    //COO
    printf("COO SpMV\n");
    int *d_row, *d_col; double *d_val, *d_vec, *d_res;
    cudaMalloc(&d_row, coo->nnz   * sizeof(int));
    cudaMalloc(&d_col, coo->nnz   * sizeof(int));
    cudaMalloc(&d_val, coo->nnz   * sizeof(double));
    cudaMalloc(&d_vec, coo->n_col * sizeof(double));
    cudaMalloc(&d_res, coo->n_row * sizeof(double));
    cudaMemcpy(d_row, coo->rows, coo->nnz   * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, coo->cols, coo->nnz   * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, coo->vals, coo->nnz   * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vector,    coo->n_col * sizeof(double), cudaMemcpyHostToDevice);
    int blocks = (coo->nnz + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int i=-WARMUP; i<NITER; i++) {
        cudaMemset(d_res, 0, coo->n_row * sizeof(double));

        TIMER_START(0);
        SpMV_COO<<<blocks, THREADS_PER_BLOCK>>>(coo->nnz, d_row, d_col, d_val, d_vec, d_res);
        cudaDeviceSynchronize();
        TIMER_STOP(0);

        double iter_time = TIMER_ELAPSED(0) / 1.e6;
        if( i >= 0) timers[i] = iter_time;

        printf("Iteration %d tooks %lfs\n", i, iter_time);
    }
    cudaFree(d_row); cudaFree(d_col); cudaFree(d_val); cudaFree(d_vec); cudaFree(d_res);
    //Sort COO
    printf("Sorted COO SpMV\n");
    TIMER_START(0);
    sort_COO(coo);
    TIMER_STOP(0);
    printf("COO sorting time: %lfs\n", TIMER_ELAPSED(0)/1.e6);
    cudaMalloc(&d_row, coo->nnz   * sizeof(int));
    cudaMalloc(&d_col, coo->nnz   * sizeof(int));
    cudaMalloc(&d_val, coo->nnz   * sizeof(double));
    cudaMalloc(&d_vec, coo->n_col * sizeof(double));
    cudaMalloc(&d_res, coo->n_row * sizeof(double));
    cudaMemcpy(d_row, coo->rows, coo->nnz   * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, coo->cols, coo->nnz   * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, coo->vals, coo->nnz   * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vector,    coo->n_col * sizeof(double), cudaMemcpyHostToDevice);
    blocks = (coo->nnz + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int i=-WARMUP; i<NITER; i++) {
        cudaMemset(d_res, 0, coo->n_row * sizeof(double));

        TIMER_START(0);
        SpMV_COO<<<blocks, THREADS_PER_BLOCK>>>(coo->nnz, d_row, d_col, d_val, d_vec, d_res);
        cudaDeviceSynchronize();
        TIMER_STOP(0);

        double iter_time = TIMER_ELAPSED(0) / 1.e6;
        if( i >= 0) timers[i] = iter_time;

        printf("Iteration %d tooks %lfs\n", i, iter_time);
    }
    cudaFree(d_row); cudaFree(d_col); cudaFree(d_val); cudaFree(d_vec); cudaFree(d_res);
    //CSR
    printf("CSR-Scalar SpMV\n");
    int *d_row_ptr, *d_col_ind; double *d_values;
    cudaMalloc(&d_row_ptr, (csr->n_row + 1) * sizeof(int));
    cudaMalloc(&d_col_ind,  csr->nnz        * sizeof(int));
    cudaMalloc(&d_values,   csr->nnz        * sizeof(double));
    cudaMalloc(&d_vec,      csr->n_col      * sizeof(double));
    cudaMalloc(&d_res,      csr->n_row      * sizeof(double));
    cudaMemcpy(d_row_ptr, csr->row_ptr, (csr->n_row + 1) * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, csr->col_ind,  csr->nnz        * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_values,  csr->values,   csr->nnz        * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec,     vector,        csr->n_col      * sizeof(double), cudaMemcpyHostToDevice);
    blocks = (csr->n_row + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    for (int i = -WARMUP; i < NITER; i++) {
        cudaMemset(d_res, 0, csr->n_row * sizeof(double));
        TIMER_START(0);
        SpMV_CSR_Scalar<<<blocks, THREADS_PER_BLOCK>>>(csr->n_row, d_row_ptr, d_col_ind, d_values, d_vec, d_res);
        cudaDeviceSynchronize();
        TIMER_STOP(0);
        double iter_time = TIMER_ELAPSED(0) / 1.e6;
        if( i >= 0) timers[i] = iter_time;
        printf("Iteration %d tooks %lfs\n", i, iter_time);
    }
    cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_values); cudaFree(d_vec); cudaFree(d_res);
    // CSR-Warp
    printf("CSR-Vector SpMV\n");
    cudaMalloc(&d_row_ptr, (csr->n_row + 1) * sizeof(int));
    cudaMalloc(&d_col_ind,  csr->nnz        * sizeof(int));
    cudaMalloc(&d_values,   csr->nnz        * sizeof(double));
    cudaMalloc(&d_vec,      csr->n_col      * sizeof(double));
    cudaMalloc(&d_res,      csr->n_row      * sizeof(double));
    cudaMemcpy(d_row_ptr, csr->row_ptr, (csr->n_row + 1) * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, csr->col_ind,  csr->nnz        * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_values,  csr->values,   csr->nnz        * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec,     vector,        csr->n_col      * sizeof(double), cudaMemcpyHostToDevice);
    blocks = (csr->n_row * WARP_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    for (int i = -WARMUP; i < NITER; i++) {
        cudaMemset(d_res, 0, csr->n_row * sizeof(double));
        TIMER_START(0);
        SpMV_CSR_Vector<<<blocks, THREADS_PER_BLOCK>>>(csr->n_row, d_row_ptr, d_col_ind, d_values, d_vec, d_res);
        cudaDeviceSynchronize();
        TIMER_STOP(0);
        double iter_time = TIMER_ELAPSED(0) / 1.e6;
        if( i >= 0) timers[i] = iter_time;
        printf("Iteration %d tooks %lfs\n", i, iter_time);

    }
    cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_values); cudaFree(d_vec); cudaFree(d_res);
    free(res);
    free(vector);
    free_CSR(csr);
    free_COO(coo);
    return 0;
}

