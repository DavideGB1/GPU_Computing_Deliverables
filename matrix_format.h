#ifndef MATRIX_FORMAT_H
#define MATRIX_FORMAT_H
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int *rows;
    int *cols;
    double *vals;
    int nnz;
    int n_row;
    int n_col;
} COO_Matrix;

typedef struct {
    int *row_ptr;
    int *col_ind;
    double *values;
    int nnz;
    int n_row;
    int n_col;
} CSR_Matrix;

COO_Matrix* create_COO(int nnz, int n_row, int n_col);
void free_COO(COO_Matrix *coo);
void sort_COO(COO_Matrix *coo);
void print_COO(COO_Matrix *coo);
CSR_Matrix* create_CSR(COO_Matrix* coo);
void free_CSR(CSR_Matrix *csr);
void print_CSR(CSR_Matrix *csr);
COO_Matrix* mm_parser(FILE *file);

#ifdef __cplusplus
}
#endif

#endif