#include "matrix.h"
#include "error.cuh"
#include "stdio.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

// common definitions to shorten code/indexing
#define BX blockIdx.x
#define BY blockIdx.y
#define TX threadIdx.x
#define TY threadIdx.y
#define DX blockDim.x 
#define DY blockDim.y
#define TIDX blockDim.x * blockIdx.x + threadIdx.x
#define TIDY blockDim.y * blockIdx.y + threadIdx.y

/*device declarations*/
template<typename T> __global__ void d_axb(const T * a, const T * b, T * c, const int ra, const int ca, const int cb);


/*template specifications*/
template __global__ void d_axb(const char * a, const char * b, char * c, const int ra, const int ca, const int cb);
template __global__ void d_axb(const int * a, const int * b, int * c, const int ra, const int ca, const int cb);
template __global__ void d_axb(const float * a, const float * b, float * c, const int ra, const int ca, const int cb);
template __global__ void d_axb(const double * a, const double * b, double * c, const int ra, const int ca, const int cb);


namespace Math
{
	template<typename T>
	void Matrix::axb_gpu(const T * a, const T * b, T * c, const int ra, const int ca, const int cb)
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		T * ad; T * bd; T * cd;

		size_t sz_a = ra * ca * sizeof(T);
		size_t sz_b = ca * cb * sizeof(T);
		size_t sz_c = ra * cb * sizeof(T);

		gpuCheckErr(cudaMalloc((void**)&ad, sz_a));
		gpuCheckErr(cudaMalloc((void**)&bd, sz_b));
		gpuCheckErr(cudaMalloc((void**)&cd, sz_c));

		gpuCheckErr(cudaMemcpy(ad, a, sz_a, cudaMemcpyHostToDevice));
		gpuCheckErr(cudaMemcpy(bd, b, sz_b, cudaMemcpyHostToDevice));
		gpuCheckErr(cudaMemcpy(cd, c, sz_c, cudaMemcpyHostToDevice));

		dim3 Threads(16, 16);
		int bx = (cb + Threads.x - 1) / Threads.x; bx = bx < 1024 ? bx : 1024;
		int by = (ra + Threads.y - 1) / Threads.y; by = by < 1024 ? by : 1024;		
		dim3 Grid(bx, by, 1);

		//printf("..dbg a(%d, %d), b(%d, %d), grid(%d,%d), threads(%d,%d)\n", ra, ca, ca, cb, bx, by, Threads.x, Threads.y);

		cudaEventRecord(start);
		d_axb<T><<<Grid, Threads>>>(ad, bd, cd, ra, ca, cb);
		gpuCheckErr(cudaPeekAtLastError());
		gpuCheckErr(cudaDeviceSynchronize());
		cudaEventRecord(stop);

		cudaMemcpy(c, cd, sz_c, cudaMemcpyDeviceToHost);
		cudaFree(ad);
		cudaFree(bd);
		cudaFree(cd);

		float ms = 0;
		cudaEventElapsedTime(&ms, start, stop);
		//printf("..gpu_mm(%3.1fms)\n", ms);
	}
}

template<typename T> __global__ void d_axb(const T * a, const T * b, T * c, const int ra, const int ca, const int cb)
{
	// note : assumed that thread indices cover matrix 
	int tx = TIDX; // col
	int ty = TIDY; // row

	if (tx >= cb || ty >= ra) return;

	const int r_ca = ca - ca / DX * DX;

	T res = 0; int num_mults = ca / DX;

	int mm = (r_ca > 0 ? num_mults + 1 : num_mults);

	int cidx = ty * cb + tx;

	for (int i = 0; i < mm; ++i)
	{
		int sa = DY * (i + ca * BY); // move to "right" in matrix "A" by 16x16 chunks 
		int sb = DX * (i * cb + BX); // move "down" matrix B by 16x16 chunks

		const T * sm_a = &(a[sa]); // collect sub-matrix of A
		const T * sm_b = &(b[sb]); // collect sub-matrix of B

		// fill one element of result matrix "c" 
		int mx = i >= num_mults ? r_ca : DX;

		int cc = ca * TY;

		for (int j = 0; j < mx; ++j)
		{
			c[cidx] += sm_a[cc + j] * sm_b[cb * j + TX];
		}
		//__syncthreads();
	}
}
