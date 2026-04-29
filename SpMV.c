
#include "matrix_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <dirent.h>

#define WARMUP 2
#define NITER 10
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256

__global__ void SpMV_CSR_Scalar(CSR_Matrix *csr, double *vector, double *res){
    //Use grid of size n_row*1
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < csr->n_row) {
        double sum = 0;
        for(int j = csr->row_ptr[thread_id]; j<csr->row_ptr[thread_id+1]; j++) {
            sum += csr->values[j]*vector[csr->col_ind[j]];
        }
        res[i] = sum;
    }
}

__global__ void SpMV_CSR_Vector(CSR_Matrix *csr, double *vector, double *res){
    //Use grid of size tbd
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = thread_id/32;
    int lane_id = thread_id%32;
    thread_id = threadIdx.x;
    __shared__ double sum[THREADS_PER_BLOCK];
    sum[thread_id] = 0;
    if (warp_id < csr->n_row){
        for(int j = csr->row_ptr[warp_id]+lane_id; j<csr->row_ptr[warp_id+1]; j+=WARP_SIZE) {
            sum[thread_id] += csr->values[j]*vector[csr->col_ind[j]];
        }
    }
     __syncthreads();
    if (lane_id <16){ sum[thread_id] += sum[thread_id+16]; }
    if (lane_id <8){ sum[thread_id] += sum[thread_id+8]; }
    if (lane_id <4){ sum[thread_id] += sum[thread_id+4]; }
    if (lane_id <2){ sum[thread_id] += sum[thread_id+2]; }
    if (lane_id ==0){
        sum[thread_id] += sum[thread_id+1];
        res[warp_id] = sum[thread_id];
    }
}

void my_SpMV(CSR_Matrix *csr, double *vector, double *res)
{
    #pragma omp parallel for
    for(int i = 0; i<csr->n_row; i++) {
        double sum = 0;
        for(int j = csr->row_ptr[i]; j<csr->row_ptr[i+1]; j++) {
            sum += csr->values[j]*vector[csr->col_ind[j]];
        }
        res[i] = sum;
    }
}
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Use: %s matrix_folder_path\n", argv[0]);
        return 1;
    }

    srand(time(NULL));
    struct dirent *entry;
    DIR *dp = opendir(argv[1]);

    if (dp == NULL) {
        perror("Error: folder not found");
        return 1;
    }

    while ((entry = readdir(dp)) != NULL) {
        if (entry->d_name[0] == '.') continue
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", argv[1], entry->d_name);
        FILE *f = fopen(path, "r");
        if (f == NULL) {
            perror("Error: file not found");
            return 1;
        }

        COO_Matrix *coo = mm_parser(f);
        fclose(f);

        CSR_Matrix *csr = create_CSR(coo);
        free_COO(coo);

        double *vector = (double * )malloc(csr->n_col*sizeof(double));
        printf("Random Vector:\n");
        for (int i = 0; i < csr->n_col; i++)
        {
            vector[i] = (double)rand()/1000000.0;
            printf("%f, ",vector[i]);
        }
        printf("\nResult Vector:\n");
        double *res = (double *)calloc(csr->n_row, sizeof(double));
        double timers[NITER];
        for (int i=-WARMUP; i<NITER; i++) {

            TIMER_START(0);
            my_SpMV(csr,vector,res);
            TIMER_STOP(0);

            double iter_time = TIMER_ELAPSED(0) / 1.e6;
            if( i >= 0) timers[i] = iter_time;

            printf("Iteration %d tooks %lfs\n", i, iter_time);
        }
        for (int i = 0; i < csr->n_row; i++)
        {
            printf("%f, ",res[i]);
        }
        printf("\n");
        free(res);
        free(vector);
        free_CSR(csr);
    }

    closedir(dp);


    return 0;
}

