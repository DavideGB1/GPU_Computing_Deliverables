#ifndef LAB3_EX1_LIB
#define LAB3_EX1_LIB

#include <sys/time.h>

#define STR(s) #s
#define XSTR(s) STR(s)

#define TIMER_DEF(n)	 struct timeval temp_1_##n={0,0}, temp_2_##n={0,0}
#define TIMER_START(n)	 gettimeofday(&temp_1_##n, (struct timezone*)0)
#define TIMER_STOP(n)	 gettimeofday(&temp_2_##n, (struct timezone*)0)
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec-temp_1_##n.tv_sec)*1.e6+(temp_2_##n.tv_usec-temp_1_##n.tv_usec))
#define TIMER_PRINT(n) \
    do { \
        int rk;\
        MPI_Comm_rank(MPI_COMM_WORLD, &rk);\
        if (rk==0) printf("Timer elapsed: %lfs\n", TIMER_ELAPSED(n)/1e6);\
        fflush(stdout);\
        sleep(0.5);\
        MPI_Barrier(MPI_COMM_WORLD);\
    } while (0);

// Put here the declaration of arithmetic_mean and geometric_mean
double geometric_mean(double array[],int len);
double arithmetic_mean(double array[],int len);
double standard_deviation(double array[],double mean, int len);
double relative_error(double cpu_result[],double gpu_result[], int len);
void print_stats(const char *label, double *timers, double *errors,int nnz, int n_row,
                 const char *matrix_name, FILE *csv, SpMVFormat format);
// ----------------------------------------------

#endif
