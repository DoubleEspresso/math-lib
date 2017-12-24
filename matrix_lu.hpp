#include "matrix.cuh"

namespace Math {
  namespace Matrix {
    
    pthread_barrier_t barrier;
	
    // LU decomposition (Doolittle algorithm .. gaussian elimination) 
    // see e.g (http://www.sci.utah.edu/~wallstedt/LU.htm)
    // input: a : square matrix
    // output: b destination matrix, lower triangular submatrix is L, upper is U
    template<typename T> 
    void lud_cpu(T * a, T * b, int * p, const int N) {
#ifdef _WIN32
      LARGE_INTEGER frequency;
      LARGE_INTEGER t1, t2;
      double elapsedTime;
      QueryPerformanceFrequency(&frequency);
      QueryPerformanceCounter(&t1);
#endif
      T * tp = new T[N*N];
      memset(tp, T(0), N*N*sizeof(T));
      memset(b, T(0), N*N*sizeof(T));
      
      // store permutations
      for (int j = 0; j < N; ++j) p[j] = j;

      // partial pivoting step
      for (int r = 0; r < N - 1; ++r) {
	T mxval = 0; int ridx = p[r];
	mostnonzero_col<T>(a, p, r, r, &mxval, &ridx, N, N);

	// found a row with an entry "farther" from 0.
	if (ridx > r) {
	  int tmp = p[r];
	  p[r] = p[ridx]; p[ridx] = tmp;
	}
      }

      for (int i = 0; i < N; ++i) {
	int piN = p[i] * N;
	int iN = i*N;

	for (int j = i; j < N; ++j) {
	  
	  T sum = 0; int jN = j*N; T _a = a[piN + j];
	  
	  for (int c = 0; c < i; ++c) _a -= b[iN + c] * tp[jN + c];
	  
	  b[iN + j] = _a;
	  tp[jN + i] = _a;
	}

	T div = b[i * N + i]; 
	if (div == 0) {
	  //printf("..small div(%3.3f)\n", div);
	  return;
	}

	for (int j = i + 1; j < N; ++j) {	
	  T sum = 0; int jN = j * N; T _a = a[p[j] * N + i];
				
	  for (int c = 0; c < i; ++c) _a -= b[jN + c] * tp[iN + c];// b[c * N + i];
				
	  b[jN + i] = _a / div;
	  tp[iN + j] = _a / div;
	}
      }
      if (tp) { delete[] tp; tp = 0; }
#ifdef _WIN32
      QueryPerformanceCounter(&t2);
      elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
      printf("..N(%d), cpu_lu(%3.1fms) (1-threads)\n", N, (float)elapsedTime);
#endif
    }
    
    template<typename T>
    struct lu_tdata {
      unsigned long tid;
      int mx_threads;
      int tstride;
      T * a;
      T * tp;
      T * b;
      int * p; // permutations
      int N;
    };

    template<typename T>
    void lud_cpup(T * a, T * b, int * p, const int N) {
      int nb_threads = 12;  // todo.
      THREAD_HANDLE * threads = new THREAD_HANDLE[nb_threads];
      std::vector<lu_tdata<T>> tdata;
      
#ifdef _WIN32
      LARGE_INTEGER frequency;
      LARGE_INTEGER t1, t2;
      double elapsedTime;
      QueryPerformanceFrequency(&frequency);
      QueryPerformanceCounter(&t1);
#endif

      pthread_barrier_init(&barrier, NULL, nb_threads);

      T * tp = new T[N*N]; memset(tp, 0., N*N * sizeof(T));
      // store permutations
      for (int j = 0; j < N; ++j) p[j] = j;

      // partial pivoting step
      for (int r = 0; r < N - 1; ++r) {
	T mxval = 0; int ridx = p[r];
	mostnonzero_col<T>(a, p, r, r, &mxval, &ridx, N, N);

	// found a row with an entry "farther" from 0.
	if (ridx > r) {
	  int tmp = p[r];
	  p[r] = p[ridx]; p[ridx] = tmp;
	}
      }

      for (int j = 0; j < nb_threads; ++j) {
	lu_tdata<T> d;
	d.mx_threads = nb_threads;
	d.tid = (unsigned long)j; d.tstride = nb_threads * 16;
	d.a = a; d.b = b; d.tp = tp; d.p = p;
	d.N = N;
	tdata.push_back(d);
      }

      for (int j = 0; j < nb_threads; ++j) threads[j] = start_thread((thread_fnc)lu_task<T>, (void*)&tdata[j], j);
      wait_threads_finish(threads, nb_threads);

      if (threads) {
	delete[] threads; threads = 0;
      }
      if (tp) { delete[] tp; tp = 0; }

#ifdef _WIN32
      QueryPerformanceCounter(&t2);
      elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
      printf("..N(%d) cpu_lu2(%3.1fms) (%d-threads)\n", N, (float)elapsedTime, nb_threads);
#endif
    }
    
    template<typename T>
    void lu_task(void * p) {
      lu_tdata<T> * d = (lu_tdata<T>*) p;
      int N = d->N;
      
      for (int i = 0; i < N; ++i) {
	int piN = d->p[i] * N;
	int iN = i * N;

	for (int j = i + d->tid; j < N; j += d->mx_threads) {
	  int jN = j * N; T _a = d->a[piN + j];

	  for (int c = 0; c < i; ++c) _a -= d->b[iN + c] * d->tp[jN + c];

	  d->b[iN + j] = _a;
	  d->tp[jN + i] = _a;
	}

	pthread_barrier_wait(&barrier);

	T div = d->b[iN + i];

	for (int j = i + 1 + d->tid; j < N; j += d->mx_threads) {
	  int jN = j*N; T _a = d->a[d->p[j] * N + i];
				
	  for (int c = 0; c < i; ++c) _a -= d->b[jN + c] * d->tp[iN + c];

	  d->b[jN + i] = _a / div;
	  d->tp[iN + j] = _a / div;
	}
	pthread_barrier_wait(&barrier);
      }
    }

    template<typename T>
    void lud_gpu(T * a, T * b, int * p, const int N) {
      lu_device<T>(a, b, p, N);
    }    
  };
};
