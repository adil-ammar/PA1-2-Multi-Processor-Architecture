#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define CHUNKSIZE 10;

typedef struct CSC{
    int nnz;
    int dim;
    double* values;
    int* r_indices;
    int* c_offset;
    double* x;
    double* y;
} CSC;

/*
Reads the input matrix file and initializes
values of the sparce matrix as -> *values 
column offset of the nnz -> *c_offset
row indices -> *r_indices
vector getting multiplied -> *x
result of multiplication -> *y
*/
void readCSC(CSC* myCSC, FILE* file){

    int idx = 0, entries, rows, cols, row, col = 0;
    double entry;

    fscanf(file, "%d %d %d", &rows, &cols, &entries);
    //printf("first row of the input matrix is %d, %d, %d\n", rows, cols, entries);
    myCSC->nnz = entries;
    myCSC->dim = rows;

    int i, j;
    double** arr = (double**)malloc(rows * sizeof(double*));

    for (i = 0; i < rows; i++){
        arr[i] = (double*)malloc(cols * sizeof(double));
        //printf("Checkpoint-x%d\n", i);
        for(j = 0; j < cols; j++){
            //printf("Checkpoint-y%d\n", j);
            arr[i][j] = 0.0;
        }
    }
    
    myCSC->values = (double*) malloc(entries * sizeof(double));
    if(myCSC->values == NULL) printf("memory for values couldn't be allocated \n");

    myCSC->r_indices = (int*) malloc(entries * sizeof(int));
    if(myCSC->r_indices == NULL) printf("memory for r_indices couldn't be allocated \n");

    myCSC->c_offset = (int*) calloc((cols+1), sizeof(int));
    if(myCSC->c_offset == NULL) printf("memory for c_offset couldn't be allocated \n");

    myCSC->x = (double*) malloc(rows * sizeof(double));
    if(myCSC->x == NULL) printf("memory for x couldn't be allocated \n");

    myCSC->y = (double*) malloc(rows * sizeof(double));
    if(myCSC->y == NULL) printf("memory for y couldn't be allocated \n");

    myCSC->c_offset[0] = 0;

    while(fscanf(file, "%d %d %lf", &row, &col, &entry) > 0){
        arr[row-1][col-1] = entry;
        myCSC->c_offset[col]++;
    }

    //initializing values and r_indices.
    int k= 0;
    for(i = 0; i < cols; i++){
        for(j = 0; j < rows; j++){

            if(arr[j][i] != 0){
                myCSC->values[k] = arr[j][i];
                myCSC->r_indices[k] = j;
                k++;
            }
        }
    }

    myCSC->c_offset[0] = 0;

    //initializing the col offset for CSC.
    for(i = 1; i <= cols; i++){
        myCSC->c_offset[i] += myCSC->c_offset[i-1];
        //printf("Current value at index %d is %d in c_offset array\n", i, myCSC->c_offset[i]);
    }

    //initializing vector x.
    for(i = 0; i < myCSC->dim; i++){
        myCSC->x[i] = 1.0;
        myCSC->y[i] = 0.0;
    }

    return;
}


double mulSequential(CSC* myCSC, int n){

    int dim = myCSC->dim, tid;
    int chunk = CHUNKSIZE;
    double* x = myCSC->x;
    double* y = myCSC->y;
    double* values = myCSC->values;
    int* c_offset = myCSC->c_offset, *r_indices = myCSC->r_indices;

    int i, j;
    
    for(i = 0; i < dim; i++)
        y[i] = 0.0;

    omp_set_num_threads(n);

    double start = omp_get_wtime();
    #pragma omp parallel shared(x, y, values, c_offset, r_indices, n) private(i, j, tid)
    {
        tid = omp_get_thread_num();

        #pragma omp for schedule(guided ,chunk)

        //multplication

        for(i = 0; i < dim; i++){
            for(j = c_offset[i]; j < c_offset[i+1]; j++){
                y[r_indices[j]] += values[j] * x[i];
            }
        }
        
    }
    double current = omp_get_wtime();

    //updating x to y
    for(i = 0; i < dim; i++)
        x[i] = y[i];

    return current - start;;
}


void printY(CSC myCSC){

    int dim = myCSC.dim, i;
    double* y = myCSC.y;
    int* c_offset = myCSC.c_offset;  

    printf("Vector Y is as follows\n");
    for(i = 0; i < dim; i++)
        printf("y[%d] = %lf \n", i, y[i]);

/*
    printf("Vector values is as follows\n");
    for(i = 0; i < myCSC.nnz; i++)
        printf("y[%d] = %lf \n", i, myCSC.values[i]);

    printf("Vector r_indices is as follows\n");
    for(i = 0; i < myCSC.nnz; i++)
        printf("y[%d] = %d \n", i, myCSC.r_indices[i]);

    printf("Vector c_offset is as follows\n");
    for(i = 0; i <= dim; i++)
        printf("y[%d] = %lf \n", i, myCSC.c_offset[i]);

    printf("Vector c_offset is as follows\n");
    for(i = 0; i <= dim; i++)
        printf("c_offset[%d] = %d \n", i, c_offset[i]);
*/
    return;
}

int main(int argc, char *argv[]){
    int tid;
    int iters = atoi(argv[2]), nthreads = atoi(argv[3]);
    struct CSC myCSC;
    
    if(argc != 4){
        printf("Please execute in the following format\n ./<executable> <matrix file> <#iterations> <#threads>\n");
        return 0;
    }

    FILE* file;
    file = fopen(argv[1], "r");
    
    //read input matrix file, allocate memory and initialize CSC struct.
    readCSC(&myCSC, file);

    //multiply "iters" number of times
    int i;
    double time;
    
    for(i = 0; i < iters; i++){
        //printf("Product of matrix and vector for iteration number: %d\n", i+1);
        time += mulSequential(&myCSC, nthreads);
        //printY(myCSC);
        //printf("End of the vector\n \n");
    }
    

    printY(myCSC);
    time /= iters;
    printf("Total time consumed in seconds: \t %f \n", time);

    free(myCSC.values);
    free(myCSC.c_offset);
    free(myCSC.r_indices);
    free(myCSC.x);
    free(myCSC.y);

    return 0;
}