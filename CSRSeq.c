#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct CSR{
    int nnz;
    int dim;
    double* values;
    int* c_indices;
    int* r_offset;
    double* x;
    double* y;
} CSR;

/*
Reads the input matrix file and initializes
values of the sparce matrix as -> *values 
column indices of the nnz -> *c_indices
row offset -> *r_offset
vector getting multiplied -> *x
result of multiplication -> *y
*/
void readCSR(CSR* myCSR, FILE* file){

    int idx = 0, entries, rows, cols, row = 0, col;
    double entry;

    fscanf(file, "%d %d %d", &rows, &cols, &entries);
    printf("first row of the input matrix is %d, %d, %d\n", rows, cols, entries);
    myCSR->nnz = entries;
    myCSR->dim = rows;
    
    myCSR->values = (double*) malloc(entries * sizeof(double));
    if(myCSR->values == NULL) printf("memory for values couldn't be allocated \n");

    myCSR->c_indices = (int*) malloc(entries * sizeof(int));
    if(myCSR->c_indices == NULL) printf("memory for c_indices couldn't be allocated \n");

    myCSR->r_offset = (int*) malloc((rows+1) * sizeof(int));
    if(myCSR->r_offset == NULL) printf("memory for r_offset couldn't be allocated \n");

    myCSR->x = (double*) malloc(rows * sizeof(double));
    if(myCSR->x == NULL) printf("memory for x couldn't be allocated \n");

    myCSR->y = (double*) malloc(rows * sizeof(double));
    if(myCSR->y == NULL) printf("memory for y couldn't be allocated \n");

    int i;
    for(i = 0; i <= myCSR->dim; i++){
        myCSR->r_offset[i] = 0;
    }

    while(fscanf(file, "%d %d %lf", &row, &col, &entry) > 0){
        //printf("current row of the input matrix is %d,%d, %lf\n", row, col, entry);
        
        //printf("Copying %lf into values array\n", entry);
    	myCSR->values[idx] = entry;
        //printf("Copying %d into c_indices array\n", col);
        myCSR->c_indices[idx++] = col-1;
        
        //printf("Copying %d into r_offset array\n", row);
        myCSR->r_offset[row]++;
    }

    //initializing the row offset for CSR.
    for(i = 1; i <= row; i++){
        myCSR->r_offset[i] += myCSR->r_offset[i-1];
        //printf("Current value at index %d is %d in r_offset array\n", i, myCSR->r_offset[i]);
    }

    //initializing vector x.
    for(i = 0; i < myCSR->dim; i++){
        myCSR->x[i] = 1.0;
        myCSR->y[i] = 0.0;
    }

    return;
}


double mulSequential(CSR* myCSR){

    int dim = myCSR->dim;
    double* x = myCSR->x;
    double* y = myCSR->y;
    double* values = myCSR->values;
    int* r_offset = myCSR->r_offset, *c_indices = myCSR->c_indices;

    int i, j;
    double start = omp_get_wtime();
    //multplication

    for(i = 0; i < dim; i++){
        y[i] = 0.0;
        for(j = r_offset[i]; j < r_offset[i+1]; j++){
            y[i] += values[j] * x[c_indices[j]];
        }
    }
    double current = omp_get_wtime();
    //updating x to y
    for(i = 0; i < dim; i++)
        x[i] = y[i];

    return current - start;
}


void printY(CSR* myCSR){

    int dim = myCSR->dim, i;
    double* y = myCSR->y;
    int* r_offset = myCSR->r_offset;

    printf("Vector Y is as follows\n");
    for(i = 0; i < dim; i++)
        printf("y[%d] = %lf \n", i, y[i]);

    /*
    printf("Vector r_offset is as follows\n");
    for(i = 0; i <= dim; i++)
        printf("r_offset[%d] = %d \n", i, r_offset[i]);
    */
    return;
}

int main(int argc, char *argv[]){
    
    if(argc != 4){
        printf("Please execute in the following format\n ./<executable> <matrix file> <#iterations> <#threads>\n");
        return 0;
    }
    
    int tid;
    int iters = atoi(argv[2]), nthreads = atoi(argv[3]);
    struct CSR myCSR;
    
    FILE* file;
    file = fopen(argv[1], "r");
    
    //read input matrix file, allocate memory and initialize CSR struct.
    readCSR(&myCSR, file);

    //multiply "iters" number of times
    int i;
    double time = 0.0;
    
    
    for(i = 0; i < iters; i++){
        //printf("Product of matrix and vector for ITERATION NUMBER: %d\n", i+1);
        time += mulSequential(&myCSR);
        //printY(&myCSR);
        //printf("End of the VECTOR\n \n");
    }
    
    printY(&myCSR);

    time /= iters;
    printf("Total time consumed in seconds: \t %f \n", time);
    free(myCSR.values);
    free(myCSR.r_offset);
    free(myCSR.c_indices);
    free(myCSR.x);
    free(myCSR.y);
    
    return 0;
}