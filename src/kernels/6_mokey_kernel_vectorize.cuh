#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    mokey_2D_blocktiling( int M, int N, int K, const uint8_t *A,
                          const uint8_t *B, float *C,
                          float *sumA, float *sumB, float *exps,
                          float b, float stdA, float meanA,
                          float stdB, float meanB) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ int8_t As[BM * BK];
  __shared__ int8_t Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in registerfile
  int8_t regM[TM] = {0.0};
  int8_t regN[TN] = {0.0};

  int count_xy[15 * TM * TN] = {0};
  int count_x[15 * TM * TN] = {0};
  int count_y[15 * TM * TN] = {0};

  int count_sign[TM * TN] = 0;
  uint8_t A_val;
  uint8_t B_val;
  uint8_t index;
  uint8_t sign;
  int sign_temp1;
  int sign_temp2;

  float t1[TM*TN] = {0.0};
  float t2[TM*TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          A_val = regM[resIdxM];
          B_val = regN[resIdxN];

          index = ((A_val & 0b01110111) + (B_val & 0b01110111));
          sign = (A_val ^ B_val);

          sign_temp1 = 1 - ((sign >> 2) & 0b10);
          sign_temp2 = 1 - ((sign >> 6) & 0b10);

          count_xy[(resIdxM * TN + resIdxN)*15 + index & 0b1111] += sign_temp;
          count_x[(resIdxM * TN + resIdxN)*15 + A_val & 0b111] += sign_temp;
          count_y[(resIdxM * TN + resIdxN)*15 + B_val & 0b111] += sign_temp;
          count_sign[(resIdxM * TN + resIdxN)] += sign_temp;

          count_xy[(resIdxM * TN + resIdxN)*15 + index >> 4] += sign_temp2;
          count_x[(resIdxM * TN + resIdxN)*15 + (A_val >> 4)] += sign_temp2;
          count_y[(resIdxM * TN + resIdxN)*15 + (B_val >> 4)] += sign_temp2;
          count_sign[(resIdxM * TN + resIdxN)] += sign_temp2;
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      for (int i = 0; i < 16; i++)
      {
        t1 += count_xy[(resIdxM * TN + resIdxN)*15 + i] * exps[i];
        t2 += count_x[(resIdxM * TN + resIdxN)*15 + i] * exps[i] + count_y[(resIdxM * TN + resIdxN)*15 + i] * exps[i];
      }
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = stdA * stdB * (t1 + b * (t2 + b * count_sign)) + sumA * meanB +
                                 sumB * meanA - len * meanA * meanB + C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}