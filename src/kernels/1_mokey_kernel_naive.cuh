#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void mokey_naive(int M, int N, int K, const uint8_t *A,
                            const uint8_t *B, float *C,
                            float sumA, float sumB, float *exps,
                            float b, float stdA, float meanA,
                            float stdB, float meanB) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // if statement is necessary to make things work under tile quantization
    if (x < M && y < N) {

        int count_xy[15] = {0};
        int count_x[15] = {0};
        int count_y[15] = {0};

        int count_sign = 0;
        uint8_t A_val;
        uint8_t B_val;
        uint8_t index;
        uint8_t sign;
        int sign_temp1;
        int sign_temp2;

        float t1 = 0;
        float t2 = 0;


        for (int i = 0; i < K; ++i) {
            A_val = A[x * K + i];
            B_val = B[i * N + y];

            index = ((A_val & 0b01110111) + (B_val & 0b01110111));
            sign = (A_val ^ B_val);

            sign_temp1 = 1 - ((sign >> 2) & 0b10);
            sign_temp2 = 1 - ((sign >> 6) & 0b10);

            count_xy[index & 0b1111] += sign_temp;
            count_x[A_val & 0b111] += sign_temp;
            count_y[B_val & 0b111] += sign_temp;
            count_sign += sign_temp;

            count_xy[index >> 4] += sign_temp2;
            count_x[(A_val >> 4)] += sign_temp2;
            count_y[(B_val >> 4)] += sign_temp2;
            count_sign += sign_temp2;
        }

        for(int i = 0; i < 16; i++) {
            t1 += count_xy[i] * exps[i];
            t2 += count_x[i] * exps[i] + count_y[i] * exps[i];
        }
        // C = α*(A@B)+β*C
        C[x * N + y] = stdA * stdB * (t1 + b * (t2 + b * count_sign)) + sumA*meanB +
                    sumB*meanA - len * meanA*meanB + C[x * N + y];
    }
}