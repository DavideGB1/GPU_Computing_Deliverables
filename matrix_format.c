#include "matrix_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

typedef struct {
    int row;
    int col;
    double val;
} SparseElement;

int compare_elements(const void *a, const void *b) {
    const SparseElement *ea = (const SparseElement *)a;
    const SparseElement *eb = (const SparseElement *)b;

    // Ordina prima per riga
    if (ea->row != eb->row) {
        return ea->row - eb->row;
    }
    // A parità di riga, ordina per colonna
    return ea->col - eb->col;
}
void sort_COO(COO_Matrix *coo) {
    if (coo == NULL || coo->nnz <= 1) return;

    // 1. Alloca un array temporaneo di struct per facilitare l'ordinamento
    SparseElement *elems = (SparseElement *)malloc(sizeof(SparseElement) * coo->nnz);
    if (elems == NULL) {
        fprintf(stderr, "Errore di allocazione memoria in sort_COO\n");
        return;
    }

    // 2. Copia i dati dagli array paralleli alla struct
    for (int i = 0; i < coo->nnz; i++) {
        elems[i].row = coo->rows[i];
        elems[i].col = coo->cols[i];
        elems[i].val = coo->vals[i];
    }

    // 3. Ordina l'array usando qsort e la tua funzione di comparazione esistente
    qsort(elems, coo->nnz, sizeof(SparseElement), compare_elements);

    // 4. Ricopia i dati ordinati negli array originali della matrice COO
    for (int i = 0; i < coo->nnz; i++) {
        coo->rows[i] = elems[i].row;
        coo->cols[i] = elems[i].col;
        coo->vals[i] = elems[i].val;
    }

    // 5. Libera la memoria temporanea
    free(elems);
}
CSR_Matrix* create_CSR(COO_Matrix* coo)
{
    CSR_Matrix *csr = (CSR_Matrix *)malloc(sizeof(CSR_Matrix));
    int *row = (int *)calloc((coo->n_row + 1), sizeof(int));
    int *col = (int *)malloc(sizeof(int) * coo->nnz);
    double *val = (double *)malloc(sizeof(double) * coo->nnz);

    // 1. Raggruppiamo i dati in un array temporaneo di struct
    SparseElement *elems = (SparseElement *)malloc(sizeof(SparseElement) * coo->nnz);
    for (int i = 0; i < coo->nnz; i++) {
        elems[i].row = coo->rows[i];
        elems[i].col = coo->cols[i];
        elems[i].val = coo->vals[i];
    }

    // 2. Usiamo il qsort nativo (velocissimo: O(N log N))
    qsort(elems, coo->nnz, sizeof(SparseElement), compare_elements);

    // 3. Costruiamo gli array CSR a partire dai dati ordinati
    for (int i = 0; i < coo->nnz; i++) {
        col[i] = elems[i].col;
        val[i] = elems[i].val;
        row[elems[i].row + 1]++;
    }

    // Accumulo per i row pointers
    row[0] = 0;
    for (int i = 0; i < coo->n_row; i++) {
        row[i+1] += row[i];
    }

    // 4. Pulizia
    free(elems);

    csr->n_row = coo->n_row;
    csr->n_col = coo->n_col;
    csr->nnz = coo->nnz;
    csr->row_ptr = row;
    csr->col_ind = col;
    csr->values = val;

    return csr;
}
void print_CSR(CSR_Matrix *csr)
{
    for (int i = 0; i < csr->n_row; ++i) {
        for (int j = csr->row_ptr[i]; j < csr->row_ptr[i+1]; ++j) {
            printf("Row: %d Col: %d Val: %f\n", i, csr->col_ind[j], csr->values[j]);
        }
    }
}
void free_CSR(CSR_Matrix *csr)
{
    free(csr->row_ptr);
    free(csr->col_ind);
    free(csr->values);
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


