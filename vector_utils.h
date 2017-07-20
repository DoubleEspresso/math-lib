#pragma once

#ifndef MATHLIB_VECTOR_UTILS_H
#define MATHLIB_VECTOR_UTILS_H

#include "threads.h"

#include <vector>
#include <iostream>

template<typename T>
inline void subvec(const T * a, T * b, const int sa, const int dim)
{
	// std::copy is actually slower
	for (int c = 0; c < dim; ++c) b[c] = a[sa + c];
}

template<typename T>
inline void _max(const T * a, T * mx, int * idx, const int N)
{
	*mx = -T(INFINITY); *idx = 0;
	for (int j = 0; j < N; ++j)
		if (a[j] > *mx) { *mx = a[j]; *idx = j; }
}

template<typename T>
inline void _min(const T * a, T * mn, int * idx, const int N)
{
	*mn = T(INFINITY); *idx = 0;
	for (int j = 0; j < N; ++j)
		if (a[j] < *mn) { *mn = a[j]; *idx = j; }
}

/*..slow parallel algo to find max*/
//template<typename T>
//struct max_t
//{
//	unsigned long tid;
//	int mx_threads;
//	int tstride;
//	const T * a;
//	T * tmp;
//	T * mx;
//	int N;
//	double r;
//};
//
//pthread_barrier_t barrier;
////MUTEX mutex;
//
//template<typename T>
//inline void _maxp(const T * a, T * mx, int * idx, const int N)
//{
//	int nb_threads = 4;
//	std::vector<max_t<T>> tdata;
//	THREAD_HANDLE * threads = new THREAD_HANDLE[nb_threads];
//
//	double r = N - N / nb_threads * nb_threads;
//	T * tmp = new T[nb_threads];
//	std::memset(tmp, -T(INFINITY), nb_threads * sizeof(T));
//	*mx = -T(INFINITY);
//
//	for (int j = 0; j < nb_threads; ++j)
//	{
//		max_t<T> m;
//		m.a = a; m.mx = mx; m.N = N;
//		m.r = r; m.tmp = tmp; m.tid = (unsigned long)j;
//		m.mx_threads = nb_threads;
//		m.tstride = nb_threads;
//		tdata.push_back(m);
//	}
//
//	pthread_barrier_init(&barrier, NULL, nb_threads);
//	//mutex_init(mutex);
//
//#ifdef _WIN32
//	LARGE_INTEGER frequency;        // ticks per second
//	LARGE_INTEGER t1, t2;           // ticks
//	double elapsedTime;
//	QueryPerformanceFrequency(&frequency);  // get ticks per second
//	QueryPerformanceCounter(&t1);	// start timer
//#endif
//
//	for (int j = 0; j < nb_threads; ++j)
//	{
//		threads[j] = start_thread((thread_fnc)max_task<T>, ((void*)&tdata[j]), j);
//	}
//	wait_threads_finish(threads, nb_threads);
//
//
//#ifdef _WIN32
//	QueryPerformanceCounter(&t2);
//	elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
//	printf("..cpup(%3.1fms) (%d-threads)\n", (float)elapsedTime, 16);
//#endif
//	printf("..maxp = %3.3f, idx=%d\n", *mx, idx);
//
//	/*cleanup*/
//	if (tmp) { delete[] tmp; tmp = 0; }
//	if (threads) { delete[] threads; threads = 0; }
//}
//
//template<typename T>
//void max_task(void * p)
//{
//	max_t<T> * d = (max_t<T>*)p;
//	int Nr = d->N - d->r;
//	T * tmp = d->tmp;
//
//	for (int j = d->tid; j < Nr; j += d->tstride)
//	{
//		if (d->a[j] > tmp[d->tid]) tmp[d->tid] = d->a[j];
//	}
//
//	pthread_barrier_wait(&barrier);
//
//	if (d->r > 0 && d->tid < d->r)
//	{
//		if (d->a[Nr + d->tid] > tmp[d->tid]) tmp[d->tid] = d->a[Nr + d->tid];
//	}
//
//	pthread_barrier_wait(&barrier);
//
//	if (d->tid == 0)
//	{
//		for (int j = 0; j < d->mx_threads; ++j)
//		{
//			if ( tmp[j] > *(d->mx)) *(d->mx) = tmp[j];
//		}
//	}
//}

#endif
