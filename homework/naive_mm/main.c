#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

void NaiveMatrixMultiply(Matrix *input0, Matrix *input1, Matrix *result)
{
    //@@ Insert code to implement naive matrix multiply here
    int rowsA= input0->shape[0];
    int colsA= input0->shape[1];
    int rowsB= input1->shape[0];
    int colsB= input1->shape[1];    
    printf("Matrix A [%d,%d], Matrix B [%d,%d]\n",rowsA,colsA,rowsB,colsB);
    for(int i=0;i<rowsA;i++) {
        for(int j=0;j<colsB;j++){               
            float v1,v2;
            //result->data[colsB] = 0;
            for(int k=0;k<rowsB;k++){ 
                v1 = input0->data[i*colsA + k];
                v2 = input1->data[k*colsB+j];      
                result->data[i*colsB +j] += v1*v2;
            }              
        }            
    }
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *input_file_c = argv[3];
    const char *input_file_d = argv[4];

    // Host input and output vectors and sizes
    Matrix host_a, host_b, host_c, answer;
    
    cl_int err;

    err = LoadMatrix(input_file_a, &host_a);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_c, &answer);
    CHECK_ERR(err, "LoadMatrix");

    int rows, cols;
    //@@ Update these values for the output rows and cols of the output
    //@@ Do not use the results from the answer matrix
    rows = host_a.shape[0];
    cols = host_b.shape[1];

    // Allocate the memory for the target.
    host_c.shape[0] = rows;
    host_c.shape[1] = cols;
    host_c.data = (float *)malloc(sizeof(float) * host_c.shape[0] * host_c.shape[1]);

    // Call your matrix multiply.
    NaiveMatrixMultiply(&host_a, &host_b, &host_c);

    // // Call to print the matrix
    PrintMatrix(&host_c);

    // Check the result of the matrix multiply
    CheckMatrix(&answer, &host_c);

    // Save the matrix
    SaveMatrix(input_file_d, &host_c);

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}