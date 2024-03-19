#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void mokey_shared_mem_block(int M, int N, int K, const uint8_t *A,
                                       const uint8_t *B, float *C,
                                       float *sumA, float *sumB, float *exps,
                                       float b, float stdA, float meanA,
                                       float stdB, float meanB)
{
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ uint8_t As[BLOCKSIZE * BLOCKSIZE];
  __shared__ uint8_t Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  // The temporary value that we're computing
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

  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
  {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
    {
      A_val = As[threadRow * BLOCKSIZE + dotIdx];
      B_val = Bs[dotIdx * BLOCKSIZE + threadCol];

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
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  for (int i = 0; i < 16; i++)
  {
    t1 += count_xy[i] * exps[i];
    t2 += count_x[i] * exps[i] + count_y[i] * exps[i];
  }
  C[threadRow * N + threadCol] = stdA * stdB * (t1 + b * (t2 + b * count_sign)) + sumA * meanB +
                                 sumB * meanA - len * meanA * meanB + C[threadRow * N + threadCol];
}