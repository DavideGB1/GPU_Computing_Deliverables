
#include "matrix_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "time/time_lib.h"
#include <dirent.h>

#define WARMUP 2
#define NITER 10

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
        fprintf(stderr, "Use: %s matrix_folder_path\n", argv[0]);
        return 1;
    }
    TIMER_DEF(0);
    srand(time(NULL));
    struct dirent *entry;
    DIR *dp = opendir(argv[1]);

    if (dp == NULL) {
        perror("Error: folder not found");
        return 1;
    }

    while ((entry = readdir(dp)) != NULL) {
        printf("Processing Matrix: %s\n", entry->d_name);

        if (entry->d_name[0] == '.') continue;
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
        printf("Random Vector Generation\n");
        for (int i = 0; i < csr->n_col; i++)
        {
            vector[i] = (double)rand()/1000000.0;
            //printf("%f, ",vector[i]);
        }
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
        printf("\n");
        free(res);
        free(vector);
        free_CSR(csr);
    }

    closedir(dp);


    return 0;
}
