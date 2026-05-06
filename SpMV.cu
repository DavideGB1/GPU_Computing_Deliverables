#include <cusparse.h>
#include "matrix_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "time/time_lib.h"
#include <dirent.h>

#define WARMUP 10
#define NITER 50
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
    if (i < nnz) {
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
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < n_row) {
        double sum = 0;
        for(int j = row_ptr[thread_id]; j < row_ptr[thread_id+1]; j++) {
            sum += values[j] * vector[col_ind[j]];
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
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = thread_id / 32;
    int lane_id = thread_id % 32;
    int local_id  = threadIdx.x;
    __shared__ double sum[THREADS_PER_BLOCK];

    sum[local_id] = 0;
    if (warp_id < n_row){
        for(int j = row_ptr[warp_id] + lane_id; j < row_ptr[warp_id+1]; j += WARP_SIZE) {
            sum[local_id] += values[j] * vector[col_ind[j]];
        }
    }
    __syncthreads();

    if (lane_id < 16) { sum[local_id] += sum[local_id + 16]; }
    __syncwarp();
    if (lane_id < 8)  { sum[local_id] += sum[local_id + 8]; }
    __syncwarp();
    if (lane_id < 4)  { sum[local_id] += sum[local_id + 4]; }
    __syncwarp();
    if (lane_id < 2)  { sum[local_id] += sum[local_id + 2]; }
    __syncwarp();

    if (lane_id == 0 && warp_id < n_row){
        sum[local_id] += sum[local_id + 1];
        res[warp_id] = sum[local_id];
    }
}

__global__ void SpMV_CSR_Vector_shuffle(
    int n_row,
    const int    *__restrict__ row_ptr,
    const int    *__restrict__ col_ind,
    const double *__restrict__ values,
    const double *__restrict__ vector,
    double *res)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = thread_id / 32;
    int lane_id = thread_id % 32;
    double sum=0.0;

    if (warp_id < n_row){
        for(int j = row_ptr[warp_id] + lane_id; j < row_ptr[warp_id+1]; j += WARP_SIZE) {
            sum += values[j] * vector[col_ind[j]];
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0 && warp_id < n_row){
        res[warp_id] = sum;
    }
}

__global__ void SpMV_ELL(
    int n_row,
	int slice_size,
    const int    *__restrict__ ell_cols,
	const int *__restrict__ slice_ptr,
    const double    *__restrict__ ell_vals,
    const double *__restrict__ vector,
    double *res)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
	int slice_idx = row / slice_size;
	int lane_id = row % slice_size;
    if (row < n_row){
	    double sum=0.0;
		int start = slice_ptr[slice_idx];
        int end   = slice_ptr[slice_idx + 1];
        int slice_width = (end - start) / slice_size;
        for(int j = 0; j < slice_width; j++) {
			int sell_id = start + (j * slice_size) + lane_id;
            sum += ell_vals[sell_id] * vector[ell_cols[sell_id]];
        }
		res[row] = sum;
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
    FILE *stats_file = fopen("stats.csv", "a");
    if (stats_file == NULL) {
        perror("Error: stats file not found\n");
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
    double *d_vec, *d_res;
    cudaMalloc(&d_vec, csr->n_col * sizeof(double));
    cudaMalloc(&d_res, csr->n_row * sizeof(double));
    cudaMemcpy(d_vec, vector, csr->n_col * sizeof(double), cudaMemcpyHostToDevice);
    printf("Vector Generation Time: %lfs\n", TIMER_ELAPSED(0)/1.e6);

    double *cpu_res = (double *)calloc(csr->n_row, sizeof(double));
    double timers[NITER];
    double errors[NITER];

    printf("Matrix: %s  rows=%d  cols=%d  nnz=%d\n\n", argv[1], csr->n_row, csr->n_col, csr->nnz);

    printf("CPU SpMV\n");
    for (int i = -WARMUP; i < NITER; i++) {
        TIMER_START(0);
        my_SpMV(csr, vector, cpu_res);
        TIMER_STOP(0);

        double iter_time = TIMER_ELAPSED(0) / 1.e3;
        if (i >= 0) timers[i] = iter_time;

        printf("Iteration %d tooks %lf ms\n", i, iter_time);
    }
    printf("\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    // COO
    printf("Sorting COO SpMV\n");
    TIMER_START(0);
    sort_COO(coo);

    TIMER_STOP(0);
    printf("COO sorting time: %lf ms\n", TIMER_ELAPSED(0)/1.e3);
    int *d_row, *d_col; double *d_val;
    double *h_res = (double*)malloc(csr->n_row * sizeof(double));
    int blocks;
    cudaMalloc(&d_row, coo->nnz   * sizeof(int));
    cudaMalloc(&d_col, coo->nnz   * sizeof(int));
    cudaMalloc(&d_val, coo->nnz   * sizeof(double));
    cudaMemcpy(d_row, coo->rows, coo->nnz   * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, coo->cols, coo->nnz   * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, coo->vals, coo->nnz   * sizeof(double), cudaMemcpyHostToDevice);
    blocks = (coo->nnz + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int i = -WARMUP; i < NITER; i++) {
        cudaMemset(d_res, 0, coo->n_row * sizeof(double));

        cudaEventRecord(start);
        SpMV_COO<<<blocks, THREADS_PER_BLOCK>>>(coo->nnz, d_row, d_col, d_val, d_vec, d_res);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        double iter_time = (double)ms;
        cudaMemcpy(h_res, d_res, coo->n_row * sizeof(double), cudaMemcpyDeviceToHost);

        if (i >= 0) {
            timers[i] = iter_time;
            double error = relative_error(cpu_res, h_res, csr->n_row);
            errors[i] = error;
        }
        printf("Iteration %d tooks %lf ms\n", i, iter_time);
    }
    cudaFree(d_row); cudaFree(d_col); cudaFree(d_val);
    free_COO(coo);
    print_stats("COO", timers, errors, coo->nnz, 0,coo->n_row, coo->n_col, argv[1], stats_file, FORMAT_COO, NITER,THREADS_PER_BLOCK);

    // CSR-Scalar
    printf("CSR-Scalar SpMV\n");
    int *d_row_ptr, *d_col_ind; double *d_values;
    cudaMalloc(&d_row_ptr, (csr->n_row + 1) * sizeof(int));
    cudaMalloc(&d_col_ind,  csr->nnz        * sizeof(int));
    cudaMalloc(&d_values,   csr->nnz        * sizeof(double));
    cudaMemcpy(d_row_ptr, csr->row_ptr, (csr->n_row + 1) * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, csr->col_ind,  csr->nnz        * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_values,  csr->values,   csr->nnz        * sizeof(double), cudaMemcpyHostToDevice);
    blocks = (csr->n_row + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int i = -WARMUP; i < NITER; i++) {
        cudaMemset(d_res, 0, csr->n_row * sizeof(double));

        cudaEventRecord(start);
        SpMV_CSR_Scalar<<<blocks, THREADS_PER_BLOCK>>>(csr->n_row, d_row_ptr, d_col_ind, d_values, d_vec, d_res);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        double iter_time = (double)ms;
        cudaMemcpy(h_res, d_res, csr->n_row * sizeof(double), cudaMemcpyDeviceToHost);

        if (i >= 0) {
            timers[i] = iter_time;
            double error = relative_error(cpu_res, h_res, csr->n_row);
            errors[i] = error;
        }
        printf("Iteration %d tooks %lf ms\n", i, iter_time);
    }
    print_stats("CSR-Scalar", timers, errors, csr->nnz, 0, csr->n_row, csr->n_col, argv[1], stats_file, FORMAT_CSR, NITER,THREADS_PER_BLOCK);

    // CSR-Vector
    printf("CSR-Vector SpMV\n");
    blocks = (csr->n_row * WARP_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int i = -WARMUP; i < NITER; i++) {
        cudaMemset(d_res, 0, csr->n_row * sizeof(double));

        cudaEventRecord(start);
        SpMV_CSR_Vector<<<blocks, THREADS_PER_BLOCK>>>(csr->n_row, d_row_ptr, d_col_ind, d_values, d_vec, d_res);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        double iter_time = (double)ms;
        cudaMemcpy(h_res, d_res, csr->n_row * sizeof(double), cudaMemcpyDeviceToHost);

        if (i >= 0) {
            timers[i] = iter_time;
            double error = relative_error(cpu_res, h_res, csr->n_row);
            errors[i] = error;
        }
        printf("Iteration %d tooks %lf ms\n", i, iter_time);
    }
    print_stats("CSR-Vector", timers, errors, csr->nnz, 0, csr->n_row, csr->n_col, argv[1], stats_file, FORMAT_CSR, NITER,THREADS_PER_BLOCK);
	// CSR-Vector Shuffle (No Shared Memory)
    printf("CSR-Vector Shuffle SpMV\n");
    blocks = (csr->n_row * WARP_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int i = -WARMUP; i < NITER; i++) {
        cudaMemset(d_res, 0, csr->n_row * sizeof(double));

        cudaEventRecord(start);
        SpMV_CSR_Vector_shuffle<<<blocks, THREADS_PER_BLOCK>>>(csr->n_row, d_row_ptr, d_col_ind, d_values, d_vec, d_res);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        double iter_time = (double)ms;
        cudaMemcpy(h_res, d_res, csr->n_row * sizeof(double), cudaMemcpyDeviceToHost);

        if (i >= 0) {
            timers[i] = iter_time;
            double error = relative_error(cpu_res, h_res, csr->n_row);
            errors[i] = error;
        }
        printf("Iteration %d tooks %lf ms\n", i, iter_time);
    }
    print_stats("CSR-Vector Shuffle", timers, errors, csr->nnz, 0, csr->n_row, csr->n_col, argv[1], stats_file, FORMAT_CSR, NITER,THREADS_PER_BLOCK);
		//Ellpack converting

	// cuSPARSE
    printf("cuSPARSE SpMV\n");

    cusparseHandle_t handle;
    cusparseCreate(&handle);
    double alpha = 1.0;
    double beta  = 0.0;

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, csr->n_col, d_vec, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, csr->n_row, d_res, CUDA_R_64F);

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, csr->n_row, csr->n_col, csr->nnz,
                      d_row_ptr, d_col_ind, d_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    size_t bufferSize = 0;
    void* d_buffer = NULL;

    cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

    cudaMalloc(&d_buffer, bufferSize);

    for (int i = -WARMUP; i < NITER; i++) {
        cudaMemset(d_res, 0, csr->n_row * sizeof(double));

        cudaEventRecord(start);
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                     CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        double iter_time = (double)ms;
        if (i >= 0) {
            timers[i] = iter_time;
            cudaMemcpy(h_res, d_res, csr->n_row * sizeof(double), cudaMemcpyDeviceToHost);
            errors[i] = relative_error(cpu_res, h_res, csr->n_row);
        }
        printf("Iteration %d took %lf ms\n", i, iter_time);
    }
    print_stats("cuSPARSE", timers, errors, csr->nnz, 0, csr->n_row, csr->n_col, argv[1], stats_file, FORMAT_CSR, NITER,THREADS_PER_BLOCK);

    cudaFree(d_buffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
	cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_values);

	long long slc_total_cells = 0;
	int n_slices = (csr->n_row + WARP_SIZE - 1) / WARP_SIZE;
    for (int i = 0; i < n_slices; i++) {
        int max_in_slice = 0;
        int row_start = i * WARP_SIZE;
        int row_end = (i + 1) * WARP_SIZE;
        if (row_end > csr->n_row) row_end = csr->n_row;

        for (int row = row_start; row < row_end; row++) {
            int row_nnz = csr->row_ptr[row + 1] - csr->row_ptr[row];
            if (row_nnz > max_in_slice) max_in_slice = row_nnz;
        }
        slc_total_cells += (long long)max_in_slice * WARP_SIZE;
    }

	unsigned long long ell_mem_required = (unsigned long long)slc_total_cells * 12 + (n_slices+1)*4;

    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);

    printf("ELL Required Memory: %zu bytes (%.2f GB)\n", ell_mem_required, (double)ell_mem_required / (1024*1024*1024));
    printf("Free VRAM: %zu bytes (%.2f GB)\n", free_byte, (double)free_byte / (1024*1024*1024));

    if (ell_mem_required > free_byte * 0.90) {
        printf("Unable to run ELL.\n");
		free_CSR(csr);
	} else {
	// ELLPACK
    printf("SELL-32 SpMV\n");
		ELL_Matrix *ell = create_ELL(csr, WARP_SIZE);
    	free_CSR(csr);

        int *ell_col, *ell_slc_ptr;
		double *ell_val;
        cudaMalloc(&ell_col, ell->max_nnz * sizeof(int));
        cudaMalloc(&ell_val, ell->max_nnz * sizeof(double));
		cudaMalloc(&ell_slc_ptr, (ell->n_slices + 1) * sizeof(int));
        cudaMemcpy(ell_col, ell->cols, ell->max_nnz * sizeof(int),    cudaMemcpyHostToDevice);
        cudaMemcpy(ell_val, ell->vals, ell->max_nnz * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(ell_slc_ptr, ell->slice_ptr, (ell->n_slices + 1) * sizeof(int), cudaMemcpyHostToDevice);
        blocks = (ell->n_row + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        for (int i = -WARMUP; i < NITER; i++) {
            cudaMemset(d_res, 0, ell->n_row * sizeof(double));

            cudaEventRecord(start);
            SpMV_ELL<<<blocks, THREADS_PER_BLOCK>>>(ell->n_row, WARP_SIZE,ell_col, ell_slc_ptr, ell_val, d_vec, d_res);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);

            double iter_time = (double)ms;
            cudaMemcpy(h_res, d_res, ell->n_row * sizeof(double), cudaMemcpyDeviceToHost);

            if (i >= 0) {
                timers[i] = iter_time;
                double error = relative_error(cpu_res, h_res, ell->n_row);
                errors[i] = error;
            }
            printf("Iteration %d tooks %lf ms\n", i, iter_time);
        }
        print_stats("SELLpack-32", timers, errors, ell->nnz, ell->max_nnz, ell->n_row, ell->n_col, argv[1], stats_file, FORMAT_ELL, NITER,THREADS_PER_BLOCK);
	    cudaFree(ell_col);
	    cudaFree(ell_slc_ptr);
	    cudaFree(ell_val);
		free_ELL(ell);
	}

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
	cudaFree(d_vec); cudaFree(d_res);


    free(cpu_res);
    free(vector);
    free(h_res);
    fclose(stats_file);
    return 0;
}