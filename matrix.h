#pragma once

#ifndef MATHLIB_MATRIX_H
#define MATHLIB_MATRIX_H

#include <stdio.h>
#include <vector>

#if defined(_MSC_VER)
  #include <Windows.h>
#endif

#include "matrix_utils.h"
#include "threads.h"
#include "complex.h"

namespace Math {
  namespace Matrix {
    
    /*matrix multiplication*/
    template<typename T> void axb_gpu(const T * a, const T * b, T * c, const int ra, const int ca, const int cb);
    template<typename T> void axb_cpu(const T * a, const T * b, T * c, const int ra, const int ca, const int cb);
    template<typename T> void axb_cpup(const T * a, const T * b, T * c, const int ra, const int ca, const int cb);
    template<typename T> void axb_ref(const T * a, const T * b, T * c, const int ra, const int ca, const int cb);
    template<typename T> void mm_task(void * p);
    
    /*LU decomposition*/
    template<typename T> void lud_cpu(T * a, T * b, int * p, const int N);
    template<typename T> void lud_cpup(T * a, T * b, int * p, const int N);
    template<typename T> void lud_gpu(T * a, T * b, int * p, const int N);
    template<typename T> void lu_task(void * p);

    /*QR decomposition*/
    template<typename T> void qr_cpu(const T * a, T * Q, T * R, const int N);
    //template<typename T> void qr_cpup(const T * a, T * Q, T * R, const int N);
    //template<typename T> void qr_gpu(const T * a, T * Q, T * R, const int N);
    //template<typename T> void qr_task(void * p);

    /*Gaussian elimination*/
    template<typename T> void ge_cpu(T * a, T * b, const int N);
    template<typename T> void ge_cpu_tri(T * a, T * b, const int N);
  };
};

#include "matrix.hpp"
#include "matrix_lu.hpp"
#include "matrix_qr.hpp"
#include "matrix_ge.hpp"

#endif
