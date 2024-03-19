#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 12) {
    std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

  // print some device info
  // CudaDeviceInfo();

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS FLOPs ceiling is reached at 8192
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  /*
  const uint8_t *A, const uint8_t *B, float *C, float *sumA, float *sumB, float *exps, float b, float stdA, float meanA, float stdB, float meanB
  */

  uint8_t *A = nullptr, *B = nullptr;
  float *C = nullptr; // host matrices

  uint8_t *dA = nullptr, *dB = nullptr;
  float *dC = nullptr; // device matrices

  float *exps = nullptr; // host vectors
  float *dExps = nullptr; // device vectors
  float sumA = 0.5, sumB = 0.5, b = 0.5, stdA = 0.5, meanA = 1, stdB = 0.5, meanB = 1; 

  A = (uint8_t *)malloc(sizeof(uint8_t) * max_size * max_size / 2); // stores 4-bit values
  B = (uint8_t *)malloc(sizeof(uint8_t) * max_size * max_size / 2);
  C = (float *)malloc(sizeof(float) * max_size * max_size);
  exps = (float *)malloc(sizeof(float) * 16);

  randomize_matrix_int(A, max_size * max_size / 2);
  randomize_matrix_int(B, max_size * max_size / 2);
  randomize_matrix(C, max_size * max_size);
  randomize_vector(exps, 16);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(uint8_t) * max_size * max_size/2));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(uint8_t) * max_size * max_size/2));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dExps, sizeof(float) * 16));

  cudaCheck(cudaMemcpy(dA, A, sizeof(uint8_t) * max_size * max_size/2,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(uint8_t) * max_size * max_size/2,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dExps, exps, sizeof(float) * 16, cudaMemcpyHostToDevice));

  int repeat_times = 20;
  for (int size : SIZE) {
    m = n = size;
    k = size/2;

    std::cout << "dimensions(m=n=2k): " << std::endl;
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0) {
      run_kernel(kernel_num, m, n, k, dA, dB, dC, sumA, sumB, dExps, b, stdA, meanA,
                 stdB, meanB); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
    }

    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      // We don't reset dC between runs to save time
      run_kernel(kernel_num, m, n, k, dA, dB, dC, sumA, sumB, dExps, b, stdA, meanA,
                 stdB, meanB); // Executes the kernel, modifies the result matrix
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);
    fflush(stdout);
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(exps);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dExps);

  return 0;
};