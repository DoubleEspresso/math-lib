#include <cstring>

#include "matrix.h"
#include "matrix_utils.h"

#include <stdio.h>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <vector>

namespace Math
{
  MUTEX mutex_mm;

  extern "C" void Matrix::axb(const char * a, const char * b, char * c, const int ra, const int ca, const int cb, bool gpu)
  {
    if (gpu) axb_gpu<char>(a, b, c, ra, ca, cb);
    else axb_cpup<char>(a, b, c, ra, ca, cb);
  }

  extern "C" void Matrix::axb(const int * a, const int * b, int * c, const int ra, const int ca, const int cb, bool gpu)
  {
    if (gpu) axb_gpu<int>(a, b, c, ra, ca, cb);
    else axb_cpup<int>(a, b, c, ra, ca, cb);
  }

  extern "C" void Matrix::axb(const float * a, const float * b, float * c, const int ra, const int ca, const int cb, bool gpu)
  {
    if (gpu) axb_gpu<float>(a, b, c, ra, ca, cb);
    else axb_cpup<float>(a, b, c, ra, ca, cb);
  }

  extern "C" void Matrix::axb(const double * a, const double * b, double * c, const int ra, const int ca, const int cb, bool gpu)
  {
    if (gpu) axb_gpu<double>(a, b, c, ra, ca, cb);
    else axb_cpup<double>(a, b, c, ra, ca, cb);	
  }

  extern "C" void Matrix::test_swap(double * a, const int ra, const int ca, const int i1, const int i2, const int nrows, const int ncols)
  {
#ifdef _WIN32
    LARGE_INTEGER frequency;        // ticks per second
    LARGE_INTEGER t1, t2;           // ticks
    double elapsedTime;
    QueryPerformanceFrequency(&frequency);  // get ticks per second
    QueryPerformanceCounter(&t1);	// start timer
#endif


    // test swap crap
    //printf("\n...before...\n");
    //for (int r = 0, idx = 0; r < ra; ++r) {
    //	for (int c = 0; c < ca; ++c, ++idx) {
    //		printf(" %3.3f ", a[idx]);
    //	}
    //	printf("\n");
    //}

    swap_rows(a, ra, ca, i1, i2);

    //printf("\n...after...\n");
    //for (int r = 0, idx = 0; r < ra; ++r) {
    //	for (int c = 0; c < ca; ++c, ++idx) {
    //		printf(" %3.3f ", a[idx]);
    //	}
    //	printf("\n");
    //}

#ifdef _WIN32
    QueryPerformanceCounter(&t2);
    elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
    printf("..swap(%3.1fms) (%d-threads)\n", (float)elapsedTime, 1);
#endif

  }

  extern "C" void Matrix::test_maxcol(const double * a, const int col, double * mx, int * idx, const int stride, const int N)
  {
#ifdef _WIN32
    LARGE_INTEGER frequency;        // ticks per second
    LARGE_INTEGER t1, t2;           // ticks
    double elapsedTime;
    QueryPerformanceFrequency(&frequency);  // get ticks per second
    QueryPerformanceCounter(&t1);	// start timer
#endif


    // test swap crap
    //printf("\n...before...\n");
    //for (int r = 0, idx = 0; r < ra; ++r) {
    //	for (int c = 0; c < ca; ++c, ++idx) {
    //		printf(" %3.3f ", a[idx]);
    //	}
    //	printf("\n");
    //}

    //swap_rows(a, ra, ca, i1, i2);
    max_col(a, col, mx, idx, stride, N);

    //printf("\n...after...\n");
    //for (int r = 0, idx = 0; r < ra; ++r) {
    //	for (int c = 0; c < ca; ++c, ++idx) {
    //		printf(" %3.3f ", a[idx]);
    //	}
    //	printf("\n");
    //}

#ifdef _WIN32
    QueryPerformanceCounter(&t2);
    elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
    printf("..max(%3.1fms) (%d-threads)\n", (float)elapsedTime, 1);
#endif

  }

  extern "C" void Matrix::axb(const Complex_c * a, const Complex_c * b, Complex_c * c, const int ra, const int ca, const int cb, bool gpu)
  {
    //if (gpu) axb_gpu<char>(a, b, c, ra, ca, cb);
    axb_cpup<Complex_c>(a, b, c, ra, ca, cb);
  }

  extern "C" void Matrix::axb(const Complex_i * a, const Complex_i * b, Complex_i * c, const int ra, const int ca, const int cb, bool gpu)
  {
    //if (gpu) axb_gpu<int>(a, b, c, ra, ca, cb);
    axb_cpup<Complex_i>(a, b, c, ra, ca, cb);
  }

  extern "C" void Matrix::axb(const Complex_f * a, const Complex_f * b, Complex_f * c, const int ra, const int ca, const int cb, bool gpu)
  {
    //if (gpu) axb_gpu<float>(a, b, c, ra, ca, cb);
    axb_cpup<Complex_f>(a, b, c, ra, ca, cb);
  }

  extern "C" void Matrix::axb(const Complex_d * a, const Complex_d * b, Complex_d * c, const int ra, const int ca, const int cb, bool gpu)
  {
    //if (gpu) axb_gpu(a, b, c, ra, ca, cb);
    axb_cpup<Complex_d>(a, b, c, ra, ca, cb);
  }
	
  /*private - utilities implementations*/	
  template<typename T>
  void Matrix::axb_cpu(const T * a, const T * b, T * c, const int ra, const int ca, const int cb)
  {
#ifdef _WIN32
    LARGE_INTEGER frequency;        // ticks per second
    LARGE_INTEGER t1, t2;           // ticks
    double elapsedTime;
    QueryPerformanceFrequency(&frequency);  // get ticks per second
    QueryPerformanceCounter(&t1);	// start timer
#endif
    const int bs = 16; // block size
    const int r_ca = ca - (ca) / bs * bs;
    const int r_ra = ra - (ra) / bs * bs;
    const int r_cb = cb - (cb) / bs * bs;
    const int r_rb = r_ca;

    //printf("rca(%d), rra(%d), rcb(%d) rrb(%d)\n", r_ca, r_ra, r_cb, r_rb);

    /* inner multiplications */
    int nmults = ca / bs;
    int row16 = ra - r_ra;
    int col16 = cb - r_cb;

    // pass 1. 16nx16n tiled multiplication
    for (int r = 0; r < row16; r += bs)
      {
	int rm = r + bs;
	T m1[256]; T m2[256]; T matres[256];

	for (int ci = 0; ci < col16; ci += bs)
	  {
	    int cm = ci + bs;

	    // inner multiplication loop
	    int sa = r * ca; int sb = ci; int mmax = nmults * bs;
	    for (int m = 0; m < mmax; m += bs)
	      {
		submat<T>(a, m1, sa + m, ca, bs, bs);
		submat<T>(b, m2, sb + cb * m, cb, 16, bs);
		strassen16x16<T>(m1, m2, matres);

		/*copy*/
		for (int ir = r, idx = 0; ir < rm; ++ir) {
		  for (int cc = ci; cc < cm; ++cc, ++idx) {
		    c[ir * cb + cc] += matres[idx];
		  }
		}
	      }

	    // if A cols do not divide evenly by 16 - do 1 more multiplication 
	    if (r_ca > 0)
	      {
		submat<T>(a, m1, r * ca + mmax, ca, bs, r_ca);
		submat<T>(b, m2, ci + cb * mmax, cb, r_rb, bs);
		axb_ref<T>(m1, m2, matres, bs, r_ca, bs); // bs x bs result (optimize?)
		/*copy*/
		for (int ir = r, idx = 0; ir < rm; ++ir) {
		  for (int cc = ci; cc < cm; ++cc, ++idx) {
		    c[ir * cb + cc] += matres[idx];
		  }
		}
	      }
	  }
      }

    // pass 2. right-most column elements (final column_dim not multiple of bs)
    if (r_cb > 0)
      {
	for (int r = 0; r < row16; r += bs)
	  {
	    int rm = r + bs;
	    // note : must have a compile time constant - we know matrix is < 256 elements
	    T m1[256]; T m2[256]; T matres[256];

	    int N = (r_ca > 0 ? ca / r_ca : 0); int r_N = (N > 0 ? ca - N * r_ca : 0);
	    int max = (N > 0 ? N : nmults); int di = (N > 0 ? r_ca : bs);
	    int sa = r * ca; int sb = cb - r_cb;

	    for (int m = 0, ii = 0; m < max; ++m, ii += di)
	      {
		submat<T>(a, m1, sa + m * di, ca, bs, di); // move to the right in A (bs x r_ca) chunks
		submat<T>(b, m2, sb + cb * ii, cb, di, r_cb); // move down B (r_ca x r_cb) chunks
		axb_ref<T>(m1, m2, matres, bs, di, r_cb); // bs x r_cb result (optimize?) 

		/*copy*/
		for (int ir = r, idx = 0; ir < rm; ++ir) { // bs
		  for (int cc = sb; cc < cb; ++cc, ++idx) { // r_cb
		    c[ir * cb + cc] += matres[idx];
		  }
		}
	      }

	    // if cols of A (or rows(b)) 
	    // do not divide evenly into 16 - do 1 more multiplication 
	    if (r_N > 0)
	      {
		submat<T>(a, m1, sa + max * di, ca, bs, r_N);
		submat<T>(b, m2, sb + cb * max * di, cb, r_N, r_cb);
		axb_ref<T>(m1, m2, matres, bs, r_N, r_cb);
		/*copy*/
		for (int ir = r, idx = 0; ir < rm; ++ir) {
		  for (int cc = sb; cc < cb; ++cc, ++idx) {
		    c[ir * cb + cc] += matres[idx];
		  }
		}
	      }
	  }
      }

    // pass 3. bottom-most row elements
    if (r_ra > 0)
      {
	for (int col = 0; col < col16; col += bs)
	  {
	    int cm = col + bs;
	    // note : must have a compile time constant - we know matrix is < 256 elements
	    T m1[256]; T m2[256]; T matres[256];

	    int sa = (ra - r_ra) * ca;
	    for (int m = 0; m < nmults; ++m)
	      {
		submat<T>(a, m1, sa + m*bs, ca, r_ra, bs); // move to the right in A (r_ra x bs) chunks
		submat<T>(b, m2, col + m*bs*cb, cb, bs, bs); // move down B (bs x bs) chunks
		axb_ref<T>(m1, m2, matres, r_ra, bs, bs); // r_ra x bs result (optimize?) 

		/*copy*/
		for (int ir = ra - r_ra, idx = 0; ir < ra; ++ir) { // r_ra
		  for (int cc = col; cc < cm; ++cc, ++idx) { // bs
		    c[ir * cb + cc] += matres[idx];
		  }
		}
	      }
	    // if cols of A (or rows of B) do not divide by 16 - do 1 more multiplication 
	    if (r_ca > 0)
	      {
		submat<T>(a, m1, sa + nmults * bs, ca, r_ra, r_ca);
		submat<T>(b, m2, col + cb * nmults * bs, cb, r_ca, bs);
		axb_ref<T>(m1, m2, matres, r_ra, r_ca, bs);

		/*copy*/
		for (int ir = ra - r_ra, idx = 0; ir < ra; ++ir) { // r_ra
		  for (int cc = col; cc < cm; ++cc, ++idx) { // bs
		    c[ir * cb + cc] += matres[idx];
		  }
		}
	      }
	  }
      }

    // 4. bottom-right corner multiplication (if cols(B) !multiples of 16)
    if (r_cb > 0 && r_ra > 0)
      {
	// note : must have a compile time constant - we know matrix is < 256 elements
	T m1[256]; T m2[256]; T matres[256];

	int sa = (ra - r_ra) * ca;
	for (int m = 0; m < nmults; ++m)
	  {
	    submat<T>(a, m1, sa + m*bs, ca, r_ra, bs); // move to the right in A (r_ra x bs) chunks
	    submat<T>(b, m2, cb - r_cb + m*bs*cb, cb, bs, r_cb); // move down B (bs x bs) chunks
	    axb_ref<T>(m1, m2, matres, r_ra, bs, r_cb); // r_ra x bs result (optimize?) 

	    /*copy*/
	    for (int ir = ra - r_ra, idx = 0; ir < ra; ++ir) { // r_ra
	      for (int cc = cb - r_cb; cc < cb; ++cc, ++idx) { // r_cb
		c[ir * cb + cc] += matres[idx];
	      }
	    }
	  }

	// if cols of A do not divide by 16 - do 1 more multiplication 
	if (r_ca > 0)
	  {
	    submat<T>(a, m1, sa + nmults * bs, ca, r_ra, r_ca);
	    submat<T>(b, m2, cb - r_cb + cb * nmults * bs, cb, r_ca, r_cb);
	    axb_ref<T>(m1, m2, matres, r_ra, r_ca, r_cb);

	    /*copy*/
	    for (int ir = ra - r_ra, idx = 0; ir < ra; ++ir) { // r_ra
	      for (int cc = cb - r_cb; cc < cb; ++cc, ++idx) { // r_cb
		c[ir * cb + cc] += matres[idx];
	      }
	    }
	  }
      }

#ifdef _WIN32
    QueryPerformanceCounter(&t2);
    elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;

    printf("..cpu(%3.1fms) (1-thread)\n", (float)elapsedTime);
#endif
  }

  template<typename T>
  void Matrix::axb_ref(const T * a, const T * b, T * c, const int ra, const int ca, const int cb)
  {
    for (int r = 0, idx = 0; r < ra; ++r)
      {
	for (int cc = 0; cc < cb; ++cc, ++idx)
	  {
	    T res = 0;
	    for (int i = 0, sidx = r*ca; i < ca; ++i) res += a[sidx + i] * b[i * cb + cc];

	    c[idx] = res;
	  }
      }
  }

  template<typename T>
  struct mm_tdata {
    unsigned long tid;
    int mx_threads;
    int tstride;
    const T * a;
    const T * b;
    T * c;
    int ra, ca, cb;
  };

  template<typename T>
  void Matrix::axb_cpup(const T * a, const T * b, T * c, const int ra, const int ca, const int cb)
  {
    int nb_threads = 12;  // todo.
    THREAD_HANDLE * threads = new THREAD_HANDLE[nb_threads];
    std::vector<mm_tdata<T>> tdata;
    memset(c, 0, sizeof(T) * ra * cb);

#ifdef _WIN32
    LARGE_INTEGER frequency;        // ticks per second
    LARGE_INTEGER t1, t2;           // ticks
    double elapsedTime;
    QueryPerformanceFrequency(&frequency);  // get ticks per second
    QueryPerformanceCounter(&t1);	// start timer
#endif
    mutex_init(mutex_mm);

    for (int j = 0; j < nb_threads; ++j)
      {
	mm_tdata<T> d;
	d.mx_threads = nb_threads;
	d.tid = (unsigned long)j; d.tstride = nb_threads * 16;
	d.a = a; d.b = b; d.c = c;
	d.ca = ca; d.cb = cb; d.ra = ra;
	tdata.push_back(d);
      }

    for (int j = 0; j < nb_threads; ++j) threads[j] = start_thread((thread_fnc)mm_task<T>, (void*)&tdata[j], j);
    wait_threads_finish(threads, nb_threads);

    if (threads) 
      {
	delete[] threads; threads = 0;
      }
#ifdef _WIN32
    QueryPerformanceCounter(&t2);
    elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
    //printf("..cpu(%3.1fms) (%d-threads)\n", (float)elapsedTime, nb_threads);
#endif
  }
	
  template<typename T>
  void Matrix::mm_task(void * p)
  {
    mm_tdata<T> * d = (mm_tdata<T>*) p;

    int bs = 16;
    int nmults = d->ca / bs;

    int r_ca = d->ca - (d->ca) / bs * bs;
    int r_ra = d->ra - (d->ra) / bs * bs;
    int r_cb = d->cb - (d->cb) / bs * bs;
    int r_rb = r_ca;
    int row16 = d->ra - r_ra;
    int col16 = d->cb - r_cb;

    // pass 1. 16nx16n matrix multiplication
    for (int r = d->tid * bs; r < row16; r += d->tstride)
      {
	T m1[256]; T m2[256]; T matres[256];
	int rm = r + bs;

	for (int ci = 0; ci < col16; ci += bs)
	  {
	    int cm = ci + bs;

	    // inner multiplication loop
	    int sa = r * d->ca; int sb = ci; int mmax = nmults * bs;
				
	    for (int m = 0; m < mmax; m += bs)
	      {
		submat<T>(d->a, m1, sa + m, d->ca, 16, 16);
		submat<T>(d->b, m2, sb + d->cb * m, d->cb, 16, 16);
		strassen16x16<T>(m1, m2, matres);
					
		/*copy*/
		for (int ir = r, idx = 0; ir < rm; ++ir) {
		  for (int cc = ci; cc < cm; ++cc, ++idx) {
		    d->c[ir * d->cb + cc] += matres[idx];
		  }
		}
	      }

	    // edge multiplication loop 
	    if (r_ca > 0 || r_cb > 0)
	      {
		submat<T>(d->a, m1, r * d->ca + mmax, d->ca, bs, r_ca);
		submat<T>(d->b, m2, ci + d->cb * mmax, d->cb, r_rb, bs);
		axb_ref<T>(m1, m2, matres, bs, r_ca, bs); // 16x16 result (optimize?)

		/*copy*/
		for (int ir = r, idx = 0; ir < rm; ++ir) {
		  for (int cc = ci; cc < cm; ++cc, ++idx) {
		    d->c[ir * d->cb + cc] += matres[idx];
		  }
		}
	      }
	  }
      }

    // pass 2. right-most column elements (final column_dim not multiple of bs)
    if (r_cb > 0)
      {
	//#pragma omp parallel for
	for (int r = d->tid * bs; r < row16; r += d->tstride)
	  {
	    int rm = r + bs;
	    // note : must have a compile time constant - we know matrix is < 256 elements
	    T m1[256]; T m2[256]; T matres[256];

	    int N = (r_ca > 0 ? d->ca / r_ca : 0); int r_N = (N > 0 ? d->ca - N * r_ca : 0);
	    int max = (N > 0 ? N : nmults); int di = (N > 0 ? r_ca : bs);
	    int sa = r * d->ca; int sb = d->cb - r_cb;

	    for (int m = 0, ii = 0; m < max; ++m, ii += di)
	      {
		submat<T>(d->a, m1, sa + m * di, d->ca, bs, di); // move to the right in A (bs x r_ca) chunks
		submat<T>(d->b, m2, sb + d->cb * ii, d->cb, di, r_cb); // move down B (r_ca x r_cb) chunks
		axb_ref<T>(m1, m2, matres, bs, di, r_cb); // bs x r_cb result (optimize?) 

		/*copy*/
		for (int ir = r, idx = 0; ir < rm; ++ir) { // bs
		  for (int cc = sb; cc < d->cb; ++cc, ++idx) { // r_cb
		    d->c[ir * d->cb + cc] += matres[idx];
		  }
		}
	      }

	    // if cols of A (or rows(b)) 
	    // do not divide evenly into 16 - do 1 more multiplication 
	    if (r_N > 0)
	      {
		submat<T>(d->a, m1, sa + max * di, d->ca, bs, r_N);
		submat<T>(d->b, m2, sb + d->cb * max * di, d->cb, r_N, r_cb);
		axb_ref<T>(m1, m2, matres, bs, r_N, r_cb);
		/*copy*/
		for (int ir = r, idx = 0; ir < rm; ++ir) {
		  for (int cc = sb; cc < d->cb; ++cc, ++idx) {
		    d->c[ir * d->cb + cc] += matres[idx];
		  }
		}
	      }
	  }
      }

    // pass 3. bottom-most row elements
    if (r_ra > 0)
      {
	//#pragma omp parallel for
	for (int col = d->tid * bs; col < col16; col += d->tstride)
	  {
	    int cm = col + bs;
	    // note : must have a compile time constant - we know matrix is < 256 elements
	    T m1[256]; T m2[256]; T matres[256];

	    int sa = (d->ra - r_ra) * d->ca;
	    for (int m = 0; m < nmults; ++m)
	      {
		submat<T>(d->a, m1, sa + m*bs, d->ca, r_ra, bs); // move to the right in A (r_ra x bs) chunks
		submat<T>(d->b, m2, col + m*bs*d->cb, d->cb, bs, bs); // move down B (bs x bs) chunks
		axb_ref<T>(m1, m2, matres, r_ra, bs, bs); // r_ra x bs result (optimize?) 

		/*copy*/
		for (int ir = d->ra - r_ra, idx = 0; ir < d->ra; ++ir) { // r_ra
		  for (int cc = col; cc < cm; ++cc, ++idx) { // bs
		    d->c[ir * d->cb + cc] += matres[idx];
		  }
		}
	      }
	    // if cols of A (or rows of B) do not divide by 16 - do 1 more multiplication 
	    if (r_ca > 0)
	      {
		submat<T>(d->a, m1, sa + nmults * bs, d->ca, r_ra, r_ca);
		submat<T>(d->b, m2, col + d->cb * nmults * bs, d->cb, r_ca, bs);
		axb_ref<T>(m1, m2, matres, r_ra, r_ca, bs);

		/*copy*/
		for (int ir = d->ra - r_ra, idx = 0; ir < d->ra; ++ir) { // r_ra
		  for (int cc = col; cc < cm; ++cc, ++idx) { // bs
		    d->c[ir * d->cb + cc] += matres[idx];
		  }
		}
	      }
	  }
      }

    // 4. bottom-right corner multiplication (if cols(B) !multiples of 16)
    if (r_cb > 0 && r_ra > 0)
      {
	// note : must have a compile time constant - we know matrix is < 256 elements
	T m1[256]; T m2[256]; T matres[256];

	int sa = (d->ra - r_ra) * d->ca;
	for (int m = d->tid; m < nmults; m += d->mx_threads)
	  {
	    submat<T>(d->a, m1, sa + m*bs, d->ca, r_ra, bs); // move to the right in A (r_ra x bs) chunks
	    submat<T>(d->b, m2, d->cb - r_cb + m * bs * d->cb, d->cb, bs, r_cb); // move down B (bs x bs) chunks
	    axb_ref<T>(m1, m2, matres, r_ra, bs, r_cb); // r_ra x bs result (optimize?) 

	    /*copy*/
	    mutex_lock(mutex_mm);
	    for (int ir = d->ra - r_ra, idx = 0; ir < d->ra; ++ir) { // r_ra
	      for (int cc = d->cb - r_cb; cc < d->cb; ++cc, ++idx) { // r_cb
		d->c[ir * d->cb + cc] += matres[idx];
	      }
	    }
	    mutex_unlock(mutex_mm);
	  }

	// if cols of A do not divide by 16 - do 1 more multiplication 
	if (r_ca > 0 && d->tid == 0)
	  {
	    submat<T>(d->a, m1, sa + nmults * bs, d->ca, r_ra, r_ca);
	    submat<T>(d->b, m2, d->cb - r_cb + d->cb * nmults * bs, d->cb, r_ca, r_cb);
	    axb_ref<T>(m1, m2, matres, r_ra, r_ca, r_cb);

	    /*copy*/
	    for (int ir = d->ra - r_ra, idx = 0; ir < d->ra; ++ir) { // r_ra
	      for (int cc = d->cb - r_cb; cc < d->cb; ++cc, ++idx) { // r_cb
		d->c[ir * d->cb + cc] += matres[idx];
	      }
	    }
	  }
      }
  }
}
