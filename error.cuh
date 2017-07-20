#pragma once
#ifndef MATHLIB_ERROR_H
#define MATHLIB_ERRORH

#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>

#define gpuCheckErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }


inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort = true) {
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
		//if (abort) exit(code);
	}
}

#endif