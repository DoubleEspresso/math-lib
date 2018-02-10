#include "vector.cuh"

namespace Math
{
  namespace Vector {
    
    template<typename T>
    void dot_cpu(const T * a, const T * b, T * c, const int dim) {
#ifdef _WIN32
      LARGE_INTEGER frequency;
      LARGE_INTEGER t1, t2;
      double elapsedTime;
      QueryPerformanceFrequency(&frequency);
      QueryPerformanceCounter(&t1);
#endif
    
      const int bs = 4; // possibly 8 with AVX-instructions.
      int r = dim - dim / bs * bs; int dim4 = dim - r;
      T dot1 = 0; T dot2 = 0; T dot3 = 0; T dot4 = 0;
      *c = 0;
    
      for (int j = 0; j < dim4; j += bs) {
	//_mm_prefetch(((char *)(a)) + 64, _MM_HINT_T0); _mm_prefetch(((char *)(b)) + 64, _MM_HINT_T0);
	dot1 += a[j] * b[j]; 
	dot2 += a[j + 1] * b[j + 1];
	dot3 += a[j + 2] * b[j + 2];
	dot4 += a[j + 3] * b[j + 3];
      }
      
      int s = dim - r;
      if (r >= 1) dot1 += a[s] * b[s];
      if (r >= 2) dot2 += a[s + 1] * b[s + 1];
      if (r >= 3) dot3 += a[s + 2] * b[s + 2];		
      (*c) += dot1 + dot2 + dot3 + dot4;

#ifdef _WIN32
      QueryPerformanceCounter(&t2);
      elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
      //printf("..cpu(%3.1fms) (1-thread)\n", (float)elapsedTime);
#endif
    }
  
    template<typename T>
    struct vd_tdata {
      MUTEX * m;
      unsigned long tid;
      int tstride;
      const T * a;
      const T * b;
      T * c;
      T dot1, dot2, dot3, dot4;
      int dim;
    };

    template<typename T>
    void dot_cpup(const T * a, const T * b, T * c, const int dim) {
      int nb_threads = 4;  // todo.
      THREAD_HANDLE * threads = new THREAD_HANDLE[nb_threads];
      std::vector< vd_tdata<T> > tdata;
      (*c) = 0;
    
#ifdef _WIN32
      LARGE_INTEGER frequency;
      LARGE_INTEGER t1, t2;
      double elapsedTime;
      QueryPerformanceFrequency(&frequency);
      QueryPerformanceCounter(&t1);
#endif
      MUTEX mutex_vec;
      mutex_init(mutex_vec);

      for (int j = 0; j < nb_threads; ++j) {
	vd_tdata<T> d;
	d.tid = (unsigned long)j; d.tstride = nb_threads * 4;
	d.a = a; d.b = b; d.dim = dim; d.c = c;
	d.m = &mutex_vec;
	tdata.push_back(d);
      }
    
      for (int j = 0; j < nb_threads; ++j) threads[j] = start_thread((thread_fnc)dot_task<T>, (void*)&tdata[j], j);
      wait_threads_finish(threads, nb_threads);    
      if (threads) { delete[] threads; threads = 0; }
    
#ifdef _WIN32
      QueryPerformanceCounter(&t2);
      elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
      printf("..cpu(%3.1fms) (%d-threads)\n", (float)elapsedTime, nb_threads);
#endif    
    }
  
    template<typename T>
    void dot_task(void * p) {
      vd_tdata<T> * d = (vd_tdata<T>*) p;
      int bs = 4;
      int r = d->dim - d->dim / bs * bs; int dim4 = d->dim - r;
      d->dot1 = 0; d->dot2 = 0; d->dot3 = 0; d->dot4 = 0;
		
      for (int j = d->tid * bs; j < dim4; j += d->tstride) {
	d->dot1 += d->a[j] * d->b[j];
	d->dot2 += d->a[j + 1] * d->b[j + 1];
	d->dot3 += d->a[j + 2] * d->b[j + 2];
	d->dot4 += d->a[j + 3] * d->b[j + 3];
      }
    
      if (r > 0 && d->tid == 0) {
	int s = d->dim - r;
	if (r == 1) d->dot1 += d->a[s] * d->b[s];
	else if (r == 2) {
	  d->dot1 += d->a[s] * d->b[s];
	  d->dot2 += d->a[s + 1] * d->b[s + 1];
	}
	else if (r == 3) {
	  d->dot1 += d->a[s] * d->b[s];
	  d->dot2 += d->a[s + 1] * d->b[s + 1];
	  d->dot3 += d->a[s + 2] * d->b[s + 2];
	}
      }

      mutex_lock(*(d->m));
      (*d->c) += d->dot1 + d->dot2 + d->dot3 + d->dot4;
      mutex_unlock(*(d->m));
    }

    template<typename T>
    void adotb_ref(const T * a, const T * b, T * c, const int dim) {
#ifdef _WIN32
      LARGE_INTEGER frequency;
      LARGE_INTEGER t1, t2;
      double elapsedTime;
      QueryPerformanceFrequency(&frequency);
      QueryPerformanceCounter(&t1);
#endif
      (*c) = 0;
      for (int j = 0; j < dim; ++j) (*c) += a[j] * b[j];
    
#ifdef _WIN32
      QueryPerformanceCounter(&t2);
      elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
      printf("..cpu_ref(%3.1fms) (1-thread)\n", (float)elapsedTime);
#endif
    }

    
    template<typename T>
    void dot_gpu(const T * a, const T * b, T * c, const int dim) {
      dot_device(a, b, c, dim);
    }
  }  
}
