#pragma once

#ifndef MATHLIB_VECTOR_UTILS_H
#define MATHLIB_VECTOR_UTILS_H

#include <vector>
#include <iostream>
#include <limits>

#include "threads.h"

template<typename T>
inline void subvec(const T * a, T * b, const int sa, const int dim) {
  // todo : bench std::copy
  for (int c = 0; c < dim; ++c) b[c] = a[sa + c];
}

template<typename T>
inline void _max(const T * a, T * mx, int * idx, const int N) {
  *mx = std::numeric_limits<T>::min();
  *idx = 0;
  for (int j = 0; j < N; ++j)
    if (a[j] > *mx) { *mx = a[j]; *idx = j; }
}

template<typename T>
inline void _min(const T * a, T * mn, int * idx, const int N) {
  *mn = std::numeric_limits<T>::max();
  *idx = 0;
  for (int j = 0; j < N; ++j)
    if (a[j] < *mn) { *mn = a[j]; *idx = j; }
}

#endif
