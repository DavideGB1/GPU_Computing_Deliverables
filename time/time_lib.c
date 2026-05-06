#include "time_lib.h"
#include <stdio.h>
#include <math.h>

// Put here the implementation of arithmetic_mean and geometric_mean
double arithmetic_mean(double array[],int len){
    double res = 0.0;
    for(int i=0;i<len;i++){
        res += array[i];
    }
    return res/len;
}
double standard_deviation(double array[],double mean, int len){
    double sum_diff = 0.0;
    for (int i = 0; i < len; i++) {
        double diff = array[i] - mean;
        sum_diff += diff * diff;
    }
    double variance = sum_diff / (len - 1);
    return sqrt(variance);
}

double relative_error(double cpu_result[],double gpu_result[], int len){

    double sum_diff = 0.0;
    for (int i = 0; i < len; i++) {
        double diff = cpu_result[i] - gpu_result[i];
        sum_diff += diff * diff;
    }
    sum_diff = sqrt(sum_diff);
    double sum_cpu = 0.0;
    for (int i = 0; i < len; i++) {
        sum_cpu += cpu_result[i] * cpu_result[i];
    }
    sum_cpu = sqrt(sum_cpu);
    return sum_diff / sum_cpu;
}

void print_stats(const char *label, double *timers, double *errors,int nnz, int max_nnz, int n_row, int n_col,
                 const char *matrix_name, FILE *csv, SpMVFormat format, int NITER, int block_size) {

    double mn = timers[0], mx = timers[0];
    for (int i = 0; i < NITER; i++) {
        if (timers[i] < mn) mn = timers[i];
        if (timers[i] > mx) mx = timers[i];
    }

    double avg = arithmetic_mean(timers, NITER);
    double stddev = standard_deviation(timers, avg, NITER);

    long bytes;
    switch (format) {
        case FORMAT_COO:
            bytes = (int)nnz * 16;
            break;
        case FORMAT_CSR:
            bytes = (int)nnz * 12 + (int)n_row * 4 + 4;
            break;
        case FORMAT_ELL:
            //High because we also account for useless 0 padding moved
            bytes = (int)max_nnz * 16;
            break;
    }
    //Vector x and Vector Res access
    bytes += (n_col + n_row) * 8;

    double gflops = (2.0 * nnz) / (avg * 1e6);
    double bandwidth = (double)bytes / (avg * 1e6);
    double avg_err = arithmetic_mean(errors, NITER);
    printf("[%s]  avg=%.6fms  stddev=%.6fms  GFLOP/s=%.2f  BW=%.2f GB/s  Avg_err=%.2e  BlockSize:%d\n",
           label, avg, stddev, gflops, bandwidth, avg_err, block_size);

    fprintf(csv, "%s,%s,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.2e,%d\n",
            matrix_name, label, avg, stddev, mn, mx, gflops, bandwidth,avg_err, block_size);
}
double geometric_mean(double array[],int len){
    double res = 1.0;
    for(int i=0;i<len;i++){
        res *= array[i] > 0.0 ? array[i] : 1;
    }
    return pow(res,1.0/len);
}
// -------------------------------------------------
