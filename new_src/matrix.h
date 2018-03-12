#pragma once

#ifndef MATHLIB_MATRIX_H
#define MATHLIB_MATRIX_H

#include <stdio.h>
#include <algorithm>

#if defined(_MSC_VER)
  #include <Windows.h>
#endif

#include "memdetail.h"

namespace Math {
  namespace Matrix {
    
    // single threaded (blocked) matrix multiplication of C = AxB
    template<typename T>
      void axb(const std::unique_ptr<T>& A,
	       const std::unique_ptr<T>& B,
	       std::unique_ptr<T>& C,
	       const unsigned int rA,
	       const unsigned int cA,
	       const unsigned int cB);
    
    // single threaded (non-blocked) matrix multiplication C = AxB
    template<typename T>
      inline void axb_naive(const std::unique_ptr<T>& A,
			    const std::unique_ptr<T>& B,
			    std::unique_ptr<T>& C,
			    const unsigned int rA,
			    const unsigned int cA,
			    const unsigned int cB);
    
  };
};

#include "matrix.hpp"

#endif
