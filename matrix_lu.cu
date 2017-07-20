#include "matrix.h"
#include "error.cuh"

#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

// common shorthands
#define BX blockIdx.x
#define BY blockIdx.y
#define TX threadIdx.x
#define TY threadIdx.y
#define DX blockDim.x 
#define DY blockDim.y
#define TIDX blockDim.x * blockIdx.x + threadIdx.x
#define TIDY blockDim.y * blockIdx.y + threadIdx.y

template<typename T> __global__ void d_lu(T * a, T * b, int * p, T * tp, const int N);

template __global__ void d_lu(double * a, double * b, int * p, double * tp, const int N);
template __global__ void d_lu(float * a, float * b, int * p, float * tp, const int N);
template __global__ void d_lu(int * a, int * b, int * p, int * tp, const int N);
//template __global__ void d_lu(const char * a, char * b, const int N);

// finds most non-zero element along a column
// accounting for possible permutations of rows and examining
// only those rows > srow
template<typename T>
inline void mostnonzero_col(const T * a, int * P, const int srow, const int scol, T * mx, int * idx, const int nrows, const int ncols)
{
	*mx = -T(INFINITY); *idx = srow;
	for (int r = srow; r < nrows; ++r) {
		T t = a[P[r] * ncols + scol];
		t = (t < 0 ? -t : t);
		if (t >(*mx)) { (*mx) = t; (*idx) = r; }
	}
}

namespace Math
{
	template<typename T>
	void Matrix::lu_gpu(T * a, T * b, int * p, const int N)
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		T * ad; T * bd; T * tp; int * pd;
		
		size_t sz = N * N * sizeof(T);

		// store permutations
		for (int j = 0; j < N; ++j) p[j] = j;

		// partial pivoting step
		for (int r = 0; r < N - 1; ++r) {
			T mxval = 0; int ridx = p[r];
			mostnonzero_col<T>(a, p, r, r, &mxval, &ridx, N, N);

			if (ridx > r) // found a row with an entry "farther" from 0.
			{
				int tmp = p[r];
				p[r] = p[ridx]; p[ridx] = tmp;
			}
		}

		gpuCheckErr(cudaMalloc((void**)&ad, sz));
		gpuCheckErr(cudaMalloc((void**)&bd, sz));
		gpuCheckErr(cudaMalloc((void**)&tp, sz));
		gpuCheckErr(cudaMalloc((void**)&pd, N * sizeof(int)));

		gpuCheckErr(cudaMemcpy(ad, a, sz, cudaMemcpyHostToDevice));
		gpuCheckErr(cudaMemcpy(bd, b, sz, cudaMemcpyHostToDevice));
		gpuCheckErr(cudaMemcpy(tp, b, sz, cudaMemcpyHostToDevice));
		gpuCheckErr(cudaMemcpy(pd, p, N*sizeof(int), cudaMemcpyHostToDevice));

		int threads = 16;
		int bs = (N + threads - 1) / threads;
		int blocks = bs < 1024 ? bs : 1024;

		//printf("..threads(%d), blocks(%d)\n", threads, blocks);
		//printf("..dbg a(%d, %d), b(%d, %d), grid(%d,%d), threads(%d,%d)\n", ra, ca, ca, cb, bx, by, Threads.x, Threads.y);

		cudaEventRecord(start);
		d_lu<T> << <blocks, threads >> >(ad, bd, pd, tp, N);
		gpuCheckErr(cudaPeekAtLastError());
		gpuCheckErr(cudaDeviceSynchronize());
		cudaEventRecord(stop);

		gpuCheckErr(cudaMemcpy(b, bd, sz, cudaMemcpyDeviceToHost));
		cudaFree(ad);
		cudaFree(bd);
		cudaFree(tp);
		cudaFree(pd);

		float ms = 0;
		cudaEventElapsedTime(&ms, start, stop);
		printf("..N(%d) gpu_lu(%3.1fms)\n", N, ms);
	}

	template<typename T> __global__ void d_lu(T * a, T * b, int * p, T * tp, const int N)
	{
		int tx = TIDX;
		int stride = gridDim.x * blockDim.x;
		
		for (int i = 0; i < N; ++i) {
			int piN = p[i] * N;
			int iN = i*N;

			for (int j = i + tx; j < N; j += stride) {
				T sum = 0; int jN = j * N; T aiNj = a[piN + j];

				for (int k = 0; k < i; ++k) aiNj -= b[iN + k] * tp[jN + k];

				b[iN + j] = aiNj;
				tp[jN + i] = aiNj;
			}
			//__syncthreads();

			T div = b[iN + i]; 
			for (int j = i + 1 + tx; j < N; j += stride) {
				int jN = j*N; T ajNi = a[p[j] * N + i];
				for (int k = 0; k < i; ++k) ajNi  -= b[jN + k] * tp[iN + k];

				b[jN + i] = ajNi / div;
				tp[iN + j] = ajNi / div;
			}
			//__syncthreads();
		}
	}
}