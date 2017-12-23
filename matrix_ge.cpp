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
  extern "C" void Matrix::ge_axb(char * a, char * b, const int N, bool tri, bool gpu) {}
  extern "C" void Matrix::ge_axb(int * a, int * b, const int N, bool tri, bool gpu) {}
  extern "C" void Matrix::ge_axb(float * a, float * b, const int N, bool tri, bool gpu) {}
  extern "C" void Matrix::ge_axb(double * a, double * b, const int N, bool tri, bool gpu)
  {
    if (tri) ge_axb_cpu_tri<double>(a, b, N);
    else ge_axb_cpu<double>(a, b, N);
  }

  // note : standard to store 'b' as the n+1-th column of matrix-a (todo)
  template<typename T>
  void Matrix::ge_axb_cpu(T * a, T * b, const int N)
  {
#ifdef _WIN32
    LARGE_INTEGER frequency;        // ticks per second
    LARGE_INTEGER t1, t2;           // ticks
    double elapsedTime;
    QueryPerformanceFrequency(&frequency);  // get ticks per second
    QueryPerformanceCounter(&t1);	// start timer
#endif

    T * at = new T[N*(N + 1)]; memset(at, T(0), N*(N + 1)*sizeof(T));
    for (int r = 0, idx = 0; r < N; ++r) {
      for (int c = 0; c < N; ++c, ++idx) {
	at[r * (N+1) + c] = a[idx];
      }
    }
    for (int j = 0, idx = N; j < N; ++j, idx += N+1) at[idx] = b[j];

    int * p = new int[N]; memset(p, 0, N * sizeof(int));

    // store permutations
    for (int j = 0; j < N; ++j) p[j] = j;

    // partial pivoting step
    for (int r = 0; r < N - 1; ++r) {
      T mxval = 0; int ridx = p[r];
      mostnonzero_col<T>(a, p, r, r, &mxval, &ridx, N, N);
      if (ridx > r) {
	int tmp = p[r];
	p[r] = p[ridx]; p[ridx] = tmp;
      }
    }

    // forward pass (elimination)
    for (int c = 0; c < N; ++c) { // columns
      int pcN = p[c] * (N + 1);

      for (int r = c+1; r < N; ++r) { // rows

	int prN = p[r]*(N+1);
	int rc = prN + c;
	T _a = at[rc] / at[pcN + c]; at[rc] = 0;

	for (int k = c + 1; k < N + 1; ++k) at[prN + k] -= _a * at[pcN + k];
      }
    }

    // backward pass (substituion)
    for (int r = N-1; r >= 0; --r) {
      int rN = p[r] * (N + 1);
      for (int c = r + 1; c < N; ++c) {
	at[rN + N] -= at[rN + c] * at[p[c]*(N+1) + N];
      }
      at[rN + N] /= at[rN + r];
    }

    // resort b-array
    for (int j = 0; j < N; ++j) {
      b[j] = at[p[j] * (N + 1) + N];
    }

    if (at) { delete[] at; at = 0; }
    if (p) { delete[] p; p = 0; }

#ifdef _WIN32
    QueryPerformanceCounter(&t2);
    elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
    printf("..cpu_ge(%3.1fms) (%d-threads)\n", (float)elapsedTime, 1);
#endif
  }

  template<typename T>
  void Matrix::ge_axb_cpu_tri(T * a, T * b, const int N)
  {
#ifdef _WIN32
    LARGE_INTEGER frequency;        // ticks per second
    LARGE_INTEGER t1, t2;           // ticks
    double elapsedTime;
    QueryPerformanceFrequency(&frequency);  // get ticks per second
    QueryPerformanceCounter(&t1);	// start timer
#endif

    T * at = new T[N*(N + 1)]; memset(at, T(0), N*(N + 1)*sizeof(T));
    for (int r = 0, idx = 0; r < N; ++r) {
      for (int c = 0; c < N; ++c, ++idx) {
	at[r * (N + 1) + c] = a[idx];
      }
    }
    for (int j = 0, idx = N; j < N; ++j, idx += N + 1) at[idx] = b[j];

    int * p = new int[N]; memset(p, 0, N * sizeof(int));

    // store permutations
    for (int j = 0; j < N; ++j) p[j] = j;

    // partial pivoting step
    for (int r = 0; r < N - 1; ++r) {
      T mxval = 0; int ridx = p[r];
      mostnonzero_col<T>(a, p, r, r, &mxval, &ridx, N, N);
      if (ridx > r) {
	int tmp = p[r];
	p[r] = p[ridx]; p[ridx] = tmp;
      }
    }

    // forward pass (elimination)
    for (int c = 0; c < N; ++c) { // columns
      int pcN = p[c] * (N + 1);

      int r = (c + 1 > N-1 ? N-1 : c + 1);
      //int mx2 = (c + 4 > N + 1 ? N + 1 : c + 4);
      //for (int r = c + 1; r < mx; ++r) { // rows

      int prN = p[r] * (N + 1);
      int rc = prN + c;
      T _a = at[rc] / at[pcN + c]; at[rc] = 0;

      for (int k = r; k < N + 1; ++k) at[prN + k] -= _a * at[pcN + k];
      //}
    }

    // backward pass (substituion)
    for (int r = N - 1; r >= 0; --r) {
      int rN = p[r] * (N + 1);
      //int mx = r + 4 > N ? N : r + 4;
      for (int c = r + 1; c < N; ++c) {
	at[rN + N] -= at[rN + c] * at[p[c] * (N + 1) + N];
      }
      at[rN + N] /= at[rN + r];
    }

    // resort b-array
    for (int j = 0; j < N; ++j) {
      b[j] = at[p[j] * (N + 1) + N];
    }

    if (at) { delete[] at; at = 0; }
    if (p) { delete[] p; p = 0; }

#ifdef _WIN32
    QueryPerformanceCounter(&t2);
    elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
    printf("..cpu_ge_tri(%3.1fms) (%d-threads)\n", (float)elapsedTime, 1);
#endif

  }

};
