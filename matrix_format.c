#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct
{
    int *rows;
    int *cols;
    double *vals;
    int nnz;
    int n_col;
    int n_row;
} COO_Matrix;
COO_Matrix* create_COO(int nnz, int n_row, int n_col)
{
    COO_Matrix *coo = (COO_Matrix *)malloc(sizeof(COO_Matrix));
    int *row = (int *)malloc(sizeof(int)*nnz);
    int *col = (int *)malloc(sizeof(int)*nnz);
    double *val = (double *)malloc(sizeof(double)*nnz);
    coo->nnz = nnz;
    coo->rows = row;
    coo->cols = col;
    coo->vals = val;
    coo->n_col = n_col;
    coo->n_row = n_row;
    return coo;
}
void print_COO(COO_Matrix *coo)
{
    for (int i = 0; i < coo->nnz; ++i)
    {
        printf("Row: %d Col: %d Val: %f\n",coo->rows[i],coo->cols[i],coo->vals[i]);
    }
}
void free_COO(COO_Matrix *coo)
{
    free(coo->rows);
    free(coo->cols);
    free(coo->vals);
    free(coo);
}

typedef struct
{
    int *rows;
    int *cols;
    double *vals;
    int nnz;
    int n_col;
    int n_row;
} CSR_Matrix;
CSR_Matrix* create_CSR(COO_Matrix* coo)
{
    CSR_Matrix *csr = (CSR_Matrix *)malloc(sizeof(CSR_Matrix));
    int *row = (int *)calloc((coo->n_row + 1), sizeof(int));
    int *col = (int *)malloc(sizeof(int)*coo->nnz);
    double *val = (double *)malloc(sizeof(double)*coo->nnz);

    int tmp;
    double tmp_val;
    int do_swap = 0;
    int n = coo->nnz;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n-i-1; j++)
        {
            do_swap = 0;
            if (
                    coo->rows[j]>coo->rows[j+1]
                    ||
                    (coo->rows[j] == coo->rows[j+1] && coo->cols[j]>coo->cols[j+1])
                )
            {
                do_swap = 1;
            }
            if (do_swap)
            {
                tmp = coo->rows[j];
                coo->rows[j] = coo->rows[j+1];
                coo->rows[j+1] = tmp;

                tmp = coo->cols[j];
                coo->cols[j] = coo->cols[j+1];
                coo->cols[j+1] = tmp;

                tmp_val = coo->vals[j];
                coo->vals[j] = coo->vals[j+1];
                coo->vals[j+1] = tmp_val;
            }
        }
    }
    for (int i = 0; i < coo->nnz; i++)
    {
        val[i] = coo->vals[i];
        col[i] = coo->cols[i];
        row[coo->rows[i] + 1]++;
    }
    row[0] = 0;
    for (int i = 0; i < coo->n_row; i++)
    {
        row[i+1] += row[i];
    }
    csr->n_row = coo->n_row;
    csr->n_col = coo->n_col;
    csr->nnz = coo->nnz;
    csr->rows= row;
    csr->cols = col;
    csr->vals = val;

    return csr;
}
void print_CSR(CSR_Matrix *csr)
{
    for (int i = 0; i < csr->n_row; ++i) {
        for (int j = csr->rows[i]; j < csr->rows[i+1]; ++j) {
            printf("Row: %d Col: %d Val: %f\n", i, csr->cols[j], csr->vals[j]);
        }
    }
}
void free_CSR(CSR_Matrix *csr)
{
    free(csr->rows);
    free(csr->cols);
    free(csr->vals);
    free(csr);
}
COO_Matrix* mm_parser(FILE *file)
{
    char buffer[256];
    int res;
    //Read header
    fgets(buffer, sizeof(buffer), file);
    int is_symmetric = 0;
    if (strstr(buffer,"symmetric")!= NULL)
    {
        is_symmetric = 1;
    }
    int is_pattern = 0;
    if (strstr(buffer,"pattern")!=NULL)
    {
        is_pattern = 1;
    }
    while (fgets(buffer, sizeof(buffer), file)) {
        if (buffer[0] != '%') break;
    }
    int M=0,N=0,nnz=0;
    sscanf(buffer,"%d %d %d",&M,&N,&nnz);
    if (is_symmetric)
    {
        nnz*=2;
    }
    COO_Matrix* coo = create_COO(nnz,M,N);

    int i=0,j=0;
    double val=1;
    for (int idx = 0; idx < nnz; idx++)
    {
        if (is_pattern)
        {
            fscanf(file, "%d %d", &i, &j);
        }else
        {
            fscanf(file, "%d %d %lf", &i, &j, &val);
        }
        i--;
        j--;
        coo->rows[idx] = i;
        coo->cols[idx] = j;
        coo->vals[idx] = val;
        if (is_symmetric)
        {
            idx++;
            coo->rows[idx] = j;
            coo->cols[idx] = i;
            coo->vals[idx] = val;
        }
    }

    return coo;
}


