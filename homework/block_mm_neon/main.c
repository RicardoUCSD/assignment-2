#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include <stdbool.h>
#include <arm_neon.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define TILE_SIZE  4

bool DEBUG_MODE = false;
#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

void BlockUnrolledNeonMatrixMultiply(Matrix *input0, Matrix *input1, Matrix *result)
{   
    unsigned int rowsA= input0->shape[0];
    unsigned int colsB= input1->shape[1]; 
    unsigned int colsA= input0->shape[1];
    unsigned int TileSize=1;

    for (unsigned int i = 1; i<TILE_SIZE; i++)
        if ((colsA % i == 0) && (colsB % i == 0))
            TileSize = i;

    unsigned int residue = (colsA) % 4; //For cases when block size is not a multiple of 4   

    for (unsigned int i = 0; i<rowsA; i+=TileSize){
        for (unsigned int j = 0; j<colsB;j+=TileSize){
            for (unsigned int blockRowGrid = i; blockRowGrid < i + TileSize; blockRowGrid++){
                for (unsigned int blockCol = j; blockCol < j+TileSize; blockCol++){
                    for (unsigned int k = 0; k<(colsA) - residue; k+=4){    //compute until we have complete blocks of 4 
                        float Ary0[4];
                        Ary0[0] = input0->data[blockRowGrid*colsA+(k+0)];
                        Ary0[1] = input0->data[blockRowGrid*colsA+(k+1)];
                        Ary0[2] = input0->data[blockRowGrid*colsA+(k+2)];
                        Ary0[3] = input0->data[blockRowGrid*colsA+(k+3)];
                        float32x4_t Q_F_vectorA = vld1q_f32(Ary0);
                    
                        float Ary1[4];
                        Ary1[0] = input1->data[(k+0)*colsB+blockCol];
                        Ary1[1] = input1->data[(k+1)*colsB+blockCol];
                        Ary1[2] = input1->data[(k+2)*colsB+blockCol];
                        Ary1[3] = input1->data[(k+3)*colsB+blockCol];
                        float32x4_t Q_F_vectorB = vld1q_f32(Ary1);
                    
                        float32x4_t data = vmulq_f32(Q_F_vectorA, Q_F_vectorB);
                        result->data[blockRowGrid*colsB+blockCol] += vaddvq_f32(data);
                    }
                    // After completed blocks have been computed, then take care of left elements
                    if (residue > 0){
                        for (unsigned int k = (colsA) - residue; k<colsA; k++){
                            float Ary0[4] = {0,0,0,0};
                            Ary0[0] = input0->data[blockRowGrid*colsA+k];
                            float32x4_t Q_F_vectorA = vld1q_f32(Ary0);

                            float Ary1[4] = {0,0,0,0};
                            Ary1[0] = input1->data[k*colsB+blockCol];
                            float32x4_t Q_F_vectorB = vld1q_f32(Ary1);

                            float32x4_t data = vmulq_f32(Q_F_vectorA, Q_F_vectorB);
                            result->data[blockRowGrid*colsB+blockCol] += vaddvq_f32(data);
                        }
                    }
                }
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
    BlockUnrolledNeonMatrixMultiply(&host_a, &host_b, &host_c);

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