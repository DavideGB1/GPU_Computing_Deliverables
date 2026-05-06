#include "matrix_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef min
#define min(a,b) (((a)<(b))?(a):(b))
#endif
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
	int idx = 0;
    for (idx = 0; idx < nnz; idx++)
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
        if (is_symmetric && (i != j))
        {
            idx++;
            coo->rows[idx] = j;
            coo->cols[idx] = i;
            coo->vals[idx] = val;
        }
    }
	coo->nnz=idx;
    return coo;
}

ELL_Matrix* create_ELL(CSR_Matrix* csr, int slice_size){
    ELL_Matrix *ell = (ELL_Matrix *)malloc(sizeof(ELL_Matrix));
	ell->C = slice_size;
	int n_slices = (csr->n_row + slice_size - 1) / slice_size;
	ell->n_slices = n_slices;
	int *slc_ptr = (int *)calloc(n_slices+1, sizeof(int));
	slc_ptr[0]=0;
	int *slc_max = (int *)calloc(n_slices, sizeof(int));
	for (int i = 0; i < n_slices; i++) {
		int max = 0;
        int val = 0;
        for (int row = i*slice_size; row < min((i+1)*slice_size, csr->n_row); row++) {
            val = csr->row_ptr[row+1]-csr->row_ptr[row];
            if (val > max) {
                max = val;
            }
        }
		slc_max[i]=max;
		slc_ptr[i+1] = max*slice_size + slc_ptr[i];
	}
	int total_size = slc_ptr[n_slices];
    int *ell_col = (int *)calloc(total_size, sizeof(int));
    double *ell_val = (double *)calloc(total_size, sizeof(double));
	for(int i = 0; i < n_slices; i++) {
		int offset = slc_ptr[i];
        int max = slc_max[i];
		for(int j = 0; j < slice_size; j++) {
			int row = (i*slice_size) + j;
			if(row < csr->n_row) {
				int csr_row_start = csr->row_ptr[row];
				int csr_row_size = csr->row_ptr[row+1] - csr_row_start;
				for(int k = 0; k < max; k++) {
					int sell_id = offset + (k*slice_size) + j;
					if(k < csr_row_size){
						ell_val[sell_id] = csr->values[csr_row_start + k];
			            ell_col[sell_id] = csr->col_ind[csr_row_start + k];

					} else {
						ell_val[sell_id] = 0.0;
			            ell_col[sell_id] = 0;
					}
				}
			}
		}
	}
	ell->slice_ptr =slc_ptr;
    ell->max_nnz = total_size;
    ell->n_row = csr->n_row;
    ell->n_col = csr->n_col;
    ell->nnz = csr->nnz;
    ell->cols = ell_col;
    ell->vals = ell_val;
	free(slc_max);
    return ell;
}
void free_ELL(ELL_Matrix *ell)
{
    free(ell->vals);
    free(ell->cols);
	free(ell->slice_ptr);
    free(ell);
}
