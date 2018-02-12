#pragma once

#ifndef MATHLIB_MATRIX2_H
#define MATHLIB_MATRIX2_H

#include <stdio.h>
#include <iostream>
#include <vector>

#if defined (_MSC_VER)
  #include <Windows.h>
#endif


// Matrix iteration class -
// purpose is to streamline/condense blocked matrix
// iteration for cache-aware matrix operations
// A matrix iterator will iterate int MxN sub-chuncks
// of a given matrix
template<typename Matrix_t>
class MatrixIterator :
public std::iterator<std::random_access_iterator_tag,
  Matrix_t,
  ptrdiff_t,
  Matrix_t*,
  Matrix_t&>
{
 public:
  explicit MatrixIterator(Matrix_t* ptr = nullptr) d_ptr(ptr) : {}
  MatrixIterator(const MatrixIterator<Matrix_t>& it) = default;
  MatrixIterator<Matrix_t>& operator=(const MatrixIterator<Matrix_t>& it) = default;  
  MatrixIterator(const MatrixIterator<Matrix_t>&& it) = default;
  MatrixIterator<Matrix_t>& operator=(cosnt MatrixIterator<Matrix_t>&& it) = default;
  ~MatrixIterator() = default;

  MatrixIterator& operator++() { /*TODO*/ }
  MatrixIterator operator++() { /*TODO*/ }
  bool operator==(iterator other) const { return false; } // TODO
  bool operator!=(iterator other) const { return !(*this == other); }
  reference operator*() const { return 0; } // TODO

  MatrixIterator begin() { /*TODO*/ return 0; }
  MatrixIterator end() { /*TODO*/ return 0; }
  
 protected:
  Matrix_t* d_ptr;
}


namespace Math {
  
}
#endif
