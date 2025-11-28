#include <cuda_runtime.h>

__global__ void log2_kernel(float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = log2f(in[idx]);
  }
}

extern "C" void log2_nvidia_batch(float *vin, float *golden, int n) {
  float *d_in, *d_out;
  cudaMalloc(&d_in, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));
  cudaMemcpy(d_in, vin, n * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  log2_kernel<<<gridSize, blockSize>>>(d_in, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(golden, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
}
