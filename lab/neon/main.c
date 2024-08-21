#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include <arm_neon.h>


#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }



int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }
    
    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *output_file = argv[3];

    // Host input and output vectors and sizes
    Matrix host_a, host_b, output;

    cl_int err;

    err = LoadMatrix(input_file_a, &host_a);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    unsigned int rows, cols;
    rows = host_a.shape[0];
    cols = host_b.shape[1];

    output.shape[0] = 1;
    output.shape[1] = 1;
    output.data = (float*)malloc(sizeof(float) * rows * cols);

    // Sum all elements of the array
    //@@ Modify the below code in the remaining demos
    unsigned int res = rows % 4;
    unsigned int placestofill = 4-res;
    unsigned int totalelems = rows + placestofill;
    //copy array to re-dimention data
    float *Ary;
    Ary = (float*)malloc(sizeof(float)*totalelems);
    memcpy(Ary, host_a.data, sizeof(float)*rows);// Copy data to temporary array in case of the array size is not multiple of 4 

    printf("\nSize of host_a.data is %d,\n\
    ...Size of Temporary Array is %d,\n\
    ...Then we have to fill with  %d places, these are zeroed posterior elements,\n\
    ...So total elements in the for-loop will be = %d\n",
             rows,
             totalelems,
             placestofill,
             totalelems);
    for(unsigned int n=0;n<placestofill && placestofill<4;n++)
    {
        printf("    ...Zeroing Extra Element number = %d\n", rows+n);
        Ary[rows+n]=0.0f;    
    }

    float sum = 0;
    for (unsigned int i = 0; i < rows * cols; (i=i+4))
    {
        float32x4_t x = vld1q_f32(Ary+i);
        sum += vaddvq_f32(x);
    }
    printf("sum: %f == %f\n\n\n", sum, host_b.data[0]);
    output.data[0] = sum;
    SaveMatrix(output_file, &output);

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(output.data);
    free(Ary);          //~Release/ flush resources to avoid memory leakages

    return 0;
}