#pragma once

#ifndef MATHLIB_MATRIX_H
#define MATHLIB_MATRIX_H

#include "complex.h"
#include "threads.h"

#ifdef MATHLIB_EXPORTS  
#define MATHLIB_API __declspec(dllexport)   
#else  
#define MATHLIB_API __declspec(dllimport)   
#endif  

/*
	todo : handle matrices smaller than 16x16, and products like 16n x m . m x 16 n where m < 16
*/

namespace Math
{
	class Matrix
	{
		template<typename T> static void axb_gpu(const T * a, const T * b, T * c, const int ra, const int ca, const int cb);
		template<typename T> static void axb_cpu(const T * a, const T * b, T * c, const int ra, const int ca, const int cb);
		template<typename T> static void axb_cpup(const T * a, const T * b, T * c, const int ra, const int ca, const int cb);
		template<typename T> static void axb_ref(const T * a, const T * b, T * c, const int ra, const int ca, const int cb);
		template<typename T> static void mm_task(void * p);
		template<typename T> static void lu_cpu(T * a, T * b, int * p, const int N);
		template<typename T> static void lu_cpup(T * a, T * b, int * p, const int N);
		template<typename T> static void lu_gpu(T * a, T * b, int * p, const int N);
		template<typename T> static void lu_task(void * p);
		template<typename T> static void qr_cpu(const T * a, T * b, const int N);
		template<typename T> static void qr_cpup(const T * a, T * b, const int N);
		template<typename T> static void qr_gpu(const T * a, T * b, const int N);
		template<typename T> static void qr_task(void * p);

	public:
		static MATHLIB_API void axb(const char * a, const char * b, char * c, const int ra, const int ca, const int cb, bool gpu = false);
		static MATHLIB_API void axb(const int * a, const int * b, int * c, const int ra, const int ca, const int cb, bool gpu = false);
		static MATHLIB_API void axb(const float * a, const float * b, float * c, const int ra, const int ca, const int cb, bool gpu = false);
		static MATHLIB_API void axb(const double * a, const double * b, double * c, const int ra, const int ca, const int cb, bool gpu = false);
		static MATHLIB_API void axb(const Complex_c * a, const Complex_c * b, Complex_c * c, const int ra, const int ca, const int cb, bool gpu = false);
		static MATHLIB_API void axb(const Complex_i * a, const Complex_i * b, Complex_i * c, const int ra, const int ca, const int cb, bool gpu = false);
		static MATHLIB_API void axb(const Complex_f * a, const Complex_f * b, Complex_f * c, const int ra, const int ca, const int cb, bool gpu = false);
		static MATHLIB_API void axb(const Complex_d * a, const Complex_d * b, Complex_d * c, const int ra, const int ca, const int cb, bool gpu = false);
		static MATHLIB_API void lu(double * a, double * b, int * p, const int N, bool gpu = false);
		static MATHLIB_API void lu(float * a, float * b, int * p, const int N, bool gpu = false);
		static MATHLIB_API void lu(int * a, int * b, int * p, const int N, bool gpu = false);
		static MATHLIB_API void lu(char * a, char * b, int * p, const int N, bool gpu = false);
		static MATHLIB_API void qr(const double * a, double * b, const int N, bool gpu = false);
		static MATHLIB_API void qr(const float * a, float * b, const int N, bool gpu = false);
		static MATHLIB_API void qr(const int * a, int * b, const int N, bool gpu = false);
		static MATHLIB_API void qr(const char * a, char * b, const int N, bool gpu = false);
		static MATHLIB_API void test_swap(double * a, const int ra, const int ca, const int i1, const int i2, const int nrows, const int ncols);
		static MATHLIB_API void test_maxcol(const double * a, const int col, double * mx, int * idx, const int stride, const int N);
	};

	/*template specifications*/
	extern "C" template void Matrix::axb_gpu(const char * a, const char * b, char * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_gpu(const int * a, const int * b, int * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_gpu(const float * a, const float * b, float * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_gpu(const double * a, const double * b, double * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpup(const char * a, const char * b, char * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpup(const int * a, const int * b, int * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpup(const float * a, const float * b, float * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpup(const double * a, const double * b, double * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpup(const Complex_c * a, const Complex_c * b, Complex_c * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpup(const Complex_i * a, const Complex_i * b, Complex_i * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpup(const Complex_f * a, const Complex_f * b, Complex_f * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpup(const Complex_d * a, const Complex_d * b, Complex_d * c, const int ra, const int ca, const int cb);	
	extern "C" template void Matrix::axb_cpu(const char * a, const char * b, char * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpu(const int * a, const int * b, int * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpu(const float * a, const float * b, float * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpu(const double * a, const double * b, double * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpu(const Complex_c * a, const Complex_c * b, Complex_c * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpu(const Complex_i * a, const Complex_i * b, Complex_i * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpu(const Complex_f * a, const Complex_f * b, Complex_f * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_cpu(const Complex_d * a, const Complex_d * b, Complex_d * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_ref(const char * a, const char * b, char * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_ref(const int * a, const int * b, int * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_ref(const float * a, const float * b, float * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_ref(const double * a, const double * b, double * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_ref(const Complex_c * a, const Complex_c * b, Complex_c * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_ref(const Complex_i * a, const Complex_i * b, Complex_i * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_ref(const Complex_f * a, const Complex_f * b, Complex_f * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::axb_ref(const Complex_d * a, const Complex_d * b, Complex_d * c, const int ra, const int ca, const int cb);
	extern "C" template void Matrix::lu_cpu(char * a, char * b, int * p, const int N);
	extern "C" template void Matrix::lu_cpu(int * a, int * b, int * p, const int N);
	extern "C" template void Matrix::lu_cpu(float * a, float * b, int * p, const int N);
	extern "C" template void Matrix::lu_cpu(double * a, double * b, int * p, const int N);
	extern "C" template void Matrix::lu_cpup(double * a, double * b, int * p, const int N);
	extern "C" template void Matrix::lu_cpup(int * a, int * b, int * p, const int N);
	extern "C" template void Matrix::lu_cpup(float * a, float * b, int * p, const int N);
	extern "C" template void Matrix::lu_cpup(double * a, double * b, int * p, const int N);
	extern "C" template void Matrix::lu_gpu(int * a, int * b, int * p, const int N);
	extern "C" template void Matrix::lu_gpu(float * a, float * b, int * p, const int N);
	extern "C" template void Matrix::lu_gpu(double * a, double * b, int * p, const int N);
	extern "C" template void Matrix::qr_cpu(const char * a, char * b, const int N);
	extern "C" template void Matrix::qr_cpu(const int * a, int * b, const int N);
	extern "C" template void Matrix::qr_cpu(const float * a, float * b, const int N);
	extern "C" template void Matrix::qr_cpup(const double * a, double * b, const int N);
	extern "C" template void Matrix::qr_cpup(const int * a, int * b, const int N);
	extern "C" template void Matrix::qr_cpup(const float * a, float * b, const int N);
	extern "C" template void Matrix::qr_cpup(const double * a, double * b, const int N);
	extern "C" template void Matrix::qr_gpu(const int * a, int * b, const int N);
	extern "C" template void Matrix::qr_gpu(const float * a, float * b, const int N);
	extern "C" template void Matrix::qr_gpu(const double * a, double * b, const int N);
}

#endif
