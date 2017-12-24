#include <cstring>
#include <vector>
#include <stdio.h>

#ifdef _WIN32
#include <Windows.h>
#endif

#include "matrix.h"
#include "matrix_utils.h"

namespace Math
{
  
  template<typename T>
  void Matrix::qr_cpu(const T * a, T * Q, T * R, const int N) {
#ifdef _WIN32
    LARGE_INTEGER frequency;
    LARGE_INTEGER t1, t2;
    double elapsedTime;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t1);
#endif

    int NN = N*N; 
    T * u = new T[N]; memset(u, 0, N*sizeof(T));
    T * l = new T[N]; memset(l, 0, N*sizeof(T));

    for (int col = 0; col < N; ++col) {
      for (int row = col + 1; row < N; ++row) {
	int rN = row*N;
	int cN = col*N;

	/* givens rotation */
	T sin = 0; T cos = 0; 
	T y = R[cN + col]; T x = R[rN + col];
	if (x > y) {
	  T ro = y / x;
	  T z = sqrt(1 + ro*ro);
	  sin = -1 / z; cos = -ro * sin;
	}
	else {
	  T ro = x / y;
	  T z = sqrt(1 + ro*ro);
	  cos = 1 / z; sin = -ro * cos;
	}
	submat(R, u, cN, N, 1, N);
	submat(R, l, rN, N, 1, N);
	for (int r = 0; r < 2; ++r) {
	  for (int c = 0; c < N; ++c) {
	    R[(r == 0 ? cN : rN) + c] = (r == 0 ? cos*u[col] - sin*l[col] : sin*u[col] + cos*l[col]);
	  }
	}
	
	if (col == 0 && row == 1) {
	  for (int j = 0, idx = 0; j < N; ++j, idx += N) Q[idx + j] = 1;
	  Q[cN + col] = cos; Q[cN + row] = -sin;
	  Q[rN + col] = sin; Q[rN + row] = cos;
	}
	else {
	  /*build Q*/
	  submat(Q, u, cN, N, 1, N);
	  submat(Q, l, rN, N, 1, N);
	  for (int r = 0; r < 2; ++r) {
	    for (int c = 0; c < N; ++c) {
	      Q[(r == 0 ? cN : rN) + c] = (r == 0 ? cos*u[col] - sin*l[col] : sin*u[col] + cos*l[col]);
	    }
	  }
	}
      }
    }

    T * tmp = new T[NN]; memset(tmp, 0, NN*sizeof(T));
    for (int r = 0, idx = 0; r < N; ++r) {
      for (int c = 0; c < N; ++c, ++idx) {
	tmp[idx] = Q[c * N + r];
      }
    }
    std::memcpy(Q, tmp, NN*sizeof(T));


#ifdef _WIN32
    QueryPerformanceCounter(&t2);
    elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
    printf("..cpu_qr(%3.1fms) (%d-threads)\n", (float)elapsedTime, 1);
#endif

    if (tmp) { delete[] tmp; tmp = 0; }
    if (l) { delete[] l; l = 0; }
    if (u) { delete[] u; u = 0; }
  }

};
