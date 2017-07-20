#pragma once

#ifndef MATHLIB_VECTOR_H
#define MATHLIB_VECTOR_H

#include "complex.h"
#include "threads.h"

#ifdef MATHLIB_EXPORTS  
#define MATHLIB_API __declspec(dllexport)   
#else  
#define MATHLIB_API __declspec(dllimport)   
#endif  

namespace Math
{
	class Vector
	{
		template<typename T> static void adotb_gpu(const T * a, const T * b, T * c, const int dim);
		template<typename T> static void adotb_cpu(const T * a, const T * b, T * c, const int dim);
		template<typename T> static void adotb_cpup(const T * a, const T * b, T * c, const int dim);
		template<typename T> static void adotb_ref(const T * a, const T * b, T * c, const int dim);
		template<typename T> static void dot_task(void * p);

	public:
		static MATHLIB_API void dot(const char * a, const char * b, char * c, const int dim, bool gpu = false);
		static MATHLIB_API void dot(const int * a, const int * b, int * c, const int dim, bool gpu = false);
		static MATHLIB_API void dot(const float * a, const float * b, float * c, const int dim, bool gpu = false);
		static MATHLIB_API void dot(const double * a, const double * b, double * c, const int dim, bool gpu = false);
		static MATHLIB_API void dot_ref(const char * a, const char * b, char * c, const int dim, bool gpu = false);
		static MATHLIB_API void dot_ref(const int * a, const int * b, int * c, const int dim, bool gpu = false);
		static MATHLIB_API void dot_ref(const float * a, const float * b, float * c, const int dim, bool gpu = false);
		static MATHLIB_API void dot_ref(const double * a, const double * b, double * c, const int dim, bool gpu = false);
	};

	/* template specifications*/
	//extern "C" template void Vector::adotb_gpu(const char * a, const char * b, char * c, const int dim);
	extern "C" template void Vector::adotb_gpu(const int * a, const int * b, int * c, const int dim);
	extern "C" template void Vector::adotb_gpu(const float * a, const float * b, float * c, const int dim);
	extern "C" template void Vector::adotb_gpu(const double * a, const double * b, double * c, const int dim);
	extern "C" template void Vector::adotb_cpu(const char * a, const char * b, char * c, const int dim);
	extern "C" template void Vector::adotb_cpu(const int * a, const int * b, int * c, const int dim);
	extern "C" template void Vector::adotb_cpu(const float * a, const float * b, float * c, const int dim);
	extern "C" template void Vector::adotb_cpu(const double * a, const double * b, double * c, const int dim);
	extern "C" template void Vector::adotb_cpup(const char * a, const char * b, char * c, const int dim);
	extern "C" template void Vector::adotb_cpup(const int * a, const int * b, int * c, const int dim);
	extern "C" template void Vector::adotb_cpup(const float * a, const float * b, float * c, const int dim);
	extern "C" template void Vector::adotb_cpup(const double * a, const double * b, double * c, const int dim);
	extern "C" template void Vector::adotb_ref(const char * a, const char * b, char * c, const int dim);
	extern "C" template void Vector::adotb_ref(const int * a, const int * b, int * c, const int dim);
	extern "C" template void Vector::adotb_ref(const float * a, const float * b, float * c, const int dim);
	extern "C" template void Vector::adotb_ref(const double * a, const double * b, double * c, const int dim);
}

#endif