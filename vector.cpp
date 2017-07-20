#include "vector.h"
#include "vector_utils.h"

#include <stdio.h>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <vector>

namespace Math
{
	MUTEX mutex_vec; 

	extern "C" void Vector::dot(const char * a, const char * b, char * c, const int dim, bool gpu)
	{
		//if (gpu) adotb_gpu(a, b, c, dim);
		adotb_cpu<char>(a, b, c, dim);
	}

	extern "C" void Vector::dot(const int * a, const int * b, int * c, const int dim, bool gpu)
	{
		//if (gpu) adotb_gpu(a, b, c, dim);
		adotb_cpu<int>(a, b, c, dim);
	}

	extern "C" void Vector::dot(const float * a, const float * b, float * c, const int dim, bool gpu)
	{
		//if (gpu) adotb_gpu(a, b, c, dim);
		adotb_cpu<float>(a, b, c, dim);
	}

	extern "C" void Vector::dot(const double * a, const double * b, double * c, const int dim, bool gpu)
	{
		if (gpu) adotb_gpu<double>(a, b, c, dim);
		else adotb_cpu<double>(a, b, c, dim);
	}

	extern "C" void Vector::dot_ref(const char * a, const char * b, char * c, const int dim, bool gpu)
	{
		adotb_ref<char>(a, b, c, dim);
	}

	extern "C" void Vector::dot_ref(const int * a, const int * b, int * c, const int dim, bool gpu)
	{
		adotb_ref<int>(a, b, c, dim);
	}

	extern "C" void Vector::dot_ref(const float * a, const float * b, float * c, const int dim, bool gpu)
	{
		adotb_ref<float>(a, b, c, dim);
	}

	extern "C" void Vector::dot_ref(const double * a, const double * b, double * c, const int dim, bool gpu)
	{
		adotb_ref<double>(a, b, c, dim);
	}

	template<typename T>
	void Vector::adotb_cpu(const T * a, const T * b, T * c, const int dim)
	{
#ifdef _WIN32
		LARGE_INTEGER frequency;        // ticks per second
		LARGE_INTEGER t1, t2;           // ticks
		double elapsedTime;
		QueryPerformanceFrequency(&frequency);  // get ticks per second
		QueryPerformanceCounter(&t1);	// start timer
#endif
		const int bs = 4; // possibly 8 with AVX-instructions.
		int r = dim - dim / bs * bs; int dim4 = dim - r;
		T dot1 = 0; T dot2 = 0; T dot3 = 0; T dot4 = 0;
		//T dot5 = 0; T dot6 = 0; T dot7 = 0; T dot8 = 0;
		*c = 0;

		//#pragma parallel for
		for (int j = 0; j < dim4; j += bs)
		{
			//_mm_prefetch(((char *)(a)) + 64, _MM_HINT_T0); _mm_prefetch(((char *)(b)) + 64, _MM_HINT_T0);
			dot1 += a[j] * b[j]; 
			dot2 += a[j + 1] * b[j + 1];
			dot3 += a[j + 2] * b[j + 2];
			dot4 += a[j + 3] * b[j + 3];
			//dot5 += a[j + 4] * b[j + 4];
			//dot6 += a[j + 5] * b[j + 5];
			//dot7 += a[j + 6] * b[j + 6];
			//dot8 += a[j + 7] * b[j + 7];
		}

		int s = dim - r;
		if (r >= 1) dot1 += a[s] * b[s];
		if (r >= 2) dot2 += a[s + 1] * b[s + 1];
		if (r >= 3) dot3 += a[s + 2] * b[s + 2];
		//if (r >= 4) dot4 += a[s + 3] * b[s + 3];
		//if (r >= 5) dot5 += a[s + 4] * b[s + 4];
		//if (r >= 6) dot6 += a[s + 5] * b[s + 5];
		//if (r >= 7) dot7 += a[s + 6] * b[s + 6];
		
		(*c) += dot1 + dot2 + dot3 + dot4;// +dot5 + dot6 + dot7 + dot8;

#ifdef _WIN32
		QueryPerformanceCounter(&t2);
		elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
		//printf("..cpu(%3.1fms) (1-thread)\n", (float)elapsedTime);
#endif
	}

	template<typename T>
	struct vd_tdata {
		unsigned long tid;
		int tstride;
		const T * a;
		const T * b;
		T * c;
		T dot1, dot2, dot3, dot4;
		int dim;
	};

	template<typename T> void Vector::adotb_cpup(const T * a, const T * b, T * c, const int dim)
	{
		int nb_threads = 4;  // todo.
		THREAD_HANDLE * threads = new THREAD_HANDLE[nb_threads];
		std::vector<vd_tdata<T>> tdata;
		(*c) = 0;

#ifdef _WIN32
		LARGE_INTEGER frequency;        // ticks per second
		LARGE_INTEGER t1, t2;           // ticks
		double elapsedTime;
		QueryPerformanceFrequency(&frequency);  // get ticks per second
		QueryPerformanceCounter(&t1);	// start timer
#endif
		mutex_init(mutex_vec);

		for (int j = 0; j < nb_threads; ++j)
		{
			vd_tdata<T> d;
			d.tid = (unsigned long)j; d.tstride = nb_threads * 4;
			d.a = a; d.b = b; d.dim = dim; d.c = c;
			tdata.push_back(d);
		}

		for (int j = 0; j < nb_threads; ++j) threads[j] = start_thread((thread_fnc)dot_task<T>, (void*)&tdata[j], j);
		wait_threads_finish(threads, nb_threads);

		if (threads)
		{
			delete[] threads; threads = 0;
		}
#ifdef _WIN32
		QueryPerformanceCounter(&t2);
		elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
		printf("..cpu(%3.1fms) (%d-threads)\n", (float)elapsedTime, nb_threads);
#endif

	}

	template<typename T>
	void Vector::dot_task(void * p)
	{
		vd_tdata<T> * d = (vd_tdata<T>*) p;

		int bs = 4;
		int r = d->dim - d->dim / bs * bs; int dim4 = d->dim - r;
		d->dot1 = 0; d->dot2 = 0; d->dot3 = 0; d->dot4 = 0;
		
		for (int j = d->tid * bs; j < dim4; j += d->tstride)
		{
			d->dot1 += d->a[j] * d->b[j];
			d->dot2 += d->a[j + 1] * d->b[j + 1];
			d->dot3 += d->a[j + 2] * d->b[j + 2];
			d->dot4 += d->a[j + 3] * d->b[j + 3];
		}

		if (r > 0 && d->tid == 0)
		{
			int s = d->dim - r;
			if (r == 1) d->dot1 += d->a[s] * d->b[s];
			else if (r == 2) { d->dot1 += d->a[s] * d->b[s]; d->dot2 += d->a[s + 1] * d->b[s + 1]; }
			else if (r == 3) { d->dot1 += d->a[s] * d->b[s]; d->dot2 += d->a[s + 1] * d->b[s + 1]; d->dot3 += d->a[s + 2] * d->b[s + 2]; }

		}

		mutex_lock(mutex_vec);
		(*d->c) += d->dot1 + d->dot2 + d->dot3 + d->dot4;
		mutex_unlock(mutex_vec);

	}

	template<typename T>
	void Vector::adotb_ref(const T * a, const T * b, T * c, const int dim)
	{
#ifdef _WIN32
		LARGE_INTEGER frequency;        // ticks per second
		LARGE_INTEGER t1, t2;           // ticks
		double elapsedTime;
		QueryPerformanceFrequency(&frequency);  // get ticks per second
		QueryPerformanceCounter(&t1);	// start timer
#endif
		(*c) = 0;
		for (int j = 0; j < dim; ++j) (*c) += a[j] * b[j];

#ifdef _WIN32
		QueryPerformanceCounter(&t2);
		elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
		printf("..cpu_ref(%3.1fms) (1-thread)\n", (float)elapsedTime);
#endif
	}
}