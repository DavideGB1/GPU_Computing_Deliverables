#ifndef LAB3_EX1_LIB
#define LAB3_EX1_LIB

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h> // Per sleep()

#ifdef __cplusplus
extern "C" {
#endif

#define STR(s) #s
#define XSTR(s) STR(s)

#define TIMER_DEF(n)     struct timeval temp_1_##n={0,0}, temp_2_##n={0,0}
#define TIMER_START(n)   gettimeofday(&temp_1_##n, (struct timezone*)0)
#define TIMER_STOP(n)    gettimeofday(&temp_2_##n, (struct timezone*)0)
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec-temp_1_##n.tv_sec)*1.e6+(temp_2_##n.tv_usec-temp_1_##n.tv_usec))

// Nota: Assicurati di includere mpi.h nel file .c/.cu o qui se usi questa macro
#define TIMER_PRINT(n) \
    do { \
        int rk;\
        MPI_Comm_rank(MPI_COMM_WORLD, &rk);\
        if (rk==0) printf("Timer elapsed: %lfs\n", TIMER_ELAPSED(n)/1e6);\
        fflush(stdout);\
        sleep(0.5);\
        MPI_Barrier(MPI_COMM_WORLD);\
    } while (0);

typedef enum { FORMAT_COO, FORMAT_CSR, FORMAT_ELL } SpMVFormat;

double geometric_mean(double array[], int len);
double arithmetic_mean(double array[], int len);
double standard_deviation(double array[], double mean, int len);
double relative_error(double cpu_result[], double gpu_result[], int len);

void print_stats(const char *label, double *timers, double *errors, int nnz, int n_row,
                 const char *matrix_name, FILE *csv, SpMVFormat format, int NITER);

#ifdef __cplusplus
} // Chiusura corretta di extern "C"
#endif

#endif // Chiusura corretta di LAB3_EX1_LIB