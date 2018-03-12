#pragma once

#ifndef MATHLIB_MEMDETAIL_H
#define MATHLIB_MEMDETAIL_H

#include <memory>
#include <algorithm>

#if defined(_MSC_VER)
  #include <Windows.h>
#endif


// c++11 specific implementations of
// make unique for generic and array template
// type parameters

template<typename T, typename... Args>
  std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
typename std::enable_if
< std::is_array<T>::value,
  std::unique_ptr<T>
  >::type
  make_unique(std::size_t n) {
  typedef typename std::remove_extent<T>::type RT;
  return std::unique_ptr<T>(new RT[n]);
}

#endif
