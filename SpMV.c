
#include "matrix_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
void * my_SpMV(CSR_Matrix *csr, double *vector, double *res)
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
        fprintf(stderr, "Use: %s <file.mtx>\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    FILE *f = fopen(argv[1], "r");
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
     my_SpMV(csr,vector,res);
    for (int i = 0; i < csr->n_row; i++)
    {
        printf("%f, ",res[i]);
    }
    printf("\n");
    free(res);
    free(vector);
    free_CSR(csr);
    return 0;
}
