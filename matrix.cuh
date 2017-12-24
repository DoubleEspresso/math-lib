#ifndef MATHLIB_MATRIX_CUH
#define MATHLIB_MATRIX_CUH

template<typename T>
void mm_device(const T * a, const T * b, T * c, const int ra, const int ca, const int cb);

template<typename T>
void lu_device(T * a, T * b, int * p, const int N);

#endif
