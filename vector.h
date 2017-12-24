#pragma once

#ifndef MATHLIB_VECTOR_H
#define MATHLIB_VECTOR_H

#include <vector>
#include <stdio.h>

#ifdef _WIN32
  #include <Windows.h>
#endif

#include "complex.h"
#include "threads.h"
#include "vector_utils.h"

namespace Math {
  namespace Vector {    
    template<typename T> void dot_gpu(const T * a, const T * b, T * c, const int dim);
    template<typename T> void dot_cpu(const T * a, const T * b, T * c, const int dim);
    template<typename T> void dot_cpup(const T * a, const T * b, T * c, const int dim);
    template<typename T> void dot_ref(const T * a, const T * b, T * c, const int dim);
    template<typename T> void dot_task(void * p);   
  };
};

#include "vector.hpp"

#endif
