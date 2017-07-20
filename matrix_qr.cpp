#include "matrix.h"
#include "matrix_utils.h"

#include <stdio.h>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <vector>

namespace Math
{
	extern "C" void Matrix::qr(const char * a, char * b, const int N, bool gpu) {}
	extern "C" void Matrix::qr(const int * a, int * b, const int N, bool gpu) {}
	extern "C" void Matrix::qr(const float * a, float * b, const int N, bool gpu) {}
	extern "C" void Matrix::qr(const double * a, double * b, const int N, bool gpu) {}


	template<typename T>
	void Matrix::qr_cpu(const T * a, T * b, const int N) {}

};