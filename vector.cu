
#include "vector.h"
#include "stdio.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

template<typename T> __global__ void dot_kernel(const T * a, const T * b, T * out, const int N);

template<typename T>
void dot_device(const T * a, const T * b, T * c, const int dim) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  T * ad; T * bd; T * cd;
  size_t bytes = sizeof(T) * dim;
  cudaMalloc((void**)&ad, bytes);
  cudaMalloc((void**)&bd, bytes);

  *c = 0;
  cudaMalloc((void**)&cd, sizeof(T));

  cudaMemcpy(ad, a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(bd, b, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(cd, c, sizeof(T), cudaMemcpyHostToDevice);

  int threads = 1024;
  int bs = (dim + threads - 1) / threads;
  int blocks = bs < 1024 ? bs : 1024;

  cudaEventRecord(start);
  dot_kernel<T> << <blocks, threads, threads * sizeof(T) >> >(ad, bd, cd, dim);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);

  cudaMemcpy(c, cd, sizeof(double), cudaMemcpyDeviceToHost);
		
  cudaFree(ad);
  cudaFree(bd);
  cudaFree(cd);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("..gpu(%3.1fms)\n", ms);
}
 
/*templated wrappers for atomic-add (no char type yet)*/
//__device__ void d_reduce(double * in, double sum) {
//  atomicAdd(in, sum);
//}

__device__ void d_reduce(float * in, float sum) {
  atomicAdd(in, sum);
}

__device__ void d_reduce(int * in, int sum) {
  atomicAdd(in, sum);
}

//__device__ void d_reduce(char * in, char sum) {
//  atomicAdd(in, sum);
//}

/*
__device__ void d_reduce(double *in, double *out, int N) {
  double sum = 0;
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  for (int i = idx, max = N / 2, stride = blockDim.x*gridDim.x; i < max; i += stride) {
    double2 val = reinterpret_cast<double2*>(in)[i];
    sum += val.x + val.y;
  }
  int i = idx + N / 2 * 2;
  if (i < N) sum += in[i];
  atomicAdd(out, sum);
}
*/

template<typename T> __global__ void dot_kernel(const T * a, const T * b, T * out, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  
  //extern __shared__ T sm[];
  extern __shared__ __align__(sizeof(T)) unsigned char tsm[];
  T * sm = reinterpret_cast<T *>(tsm);
  
  sm[threadIdx.x] = 0; // init
  
  for (int i = tid; i < N; i += stride) sm[threadIdx.x] += a[i] * b[i];
  
  __syncthreads();  
  T sum = 0;
  sum += sm[threadIdx.x];  
  __syncthreads();  
  d_reduce(out, sum); // TODO
}

// -------------------------------
// template specializations
// -------------------------------
//__device__ void d_reduce(double *in, double *out, int N);
//__device__ void d_reduce(double *in, double out);
__device__ void d_reduce(float *in, float out);
__device__ void d_reduce(int *in, int out);
//__device__ void d_reduce(char *in, char out);

//template void dot_device(const char * a, const char * b, char * c, const int dim);
template void dot_device(const int * a, const int * b, int * c, const int dim);
template void dot_device(const float * a, const float * b, float * c, const int dim);
//template void dot_device(const double * a, const double * b, double * c, const int dim);

//template __global__ void dot_kernel(const char * a, const char * b, char * out, const int N);
template __global__ void dot_kernel(const int * a, const int * b, int * out, const int N);
template __global__ void dot_kernel(const float * a, const float * b, float * out, const int N);
//template __global__ void dot_kernel(const double * a, const double * b, double * out, const int N);

