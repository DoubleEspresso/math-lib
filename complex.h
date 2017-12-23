#pragma once

#ifndef MATHLIB_COMPLEX_UTILS_H
#define MATHLIB_COMPLEX_UTILS_H

#include <cmath>

template<typename T>
class Complex {
 public:
  T re, im;
  
 Complex() : re(0), im(0) {};
 Complex(const T r, const T i) : re(r), im(i) {};
 Complex(const Complex<T>& o) : re(o.re), im(o.im) {};
 Complex(const T& o) : re(o), im(0) {};
  ~Complex() {};

  Complex operator=(const Complex& other) {
    im = other.im; re = other.re;
    return *this;
  }

  Complex conj() { return Complex<T>(re, -im); }
  inline T abs2() { return T(re*re + im*im); }
  inline T abs() { return T(sqrt(abs2())); }
  Complex pow(const double& p);

  Complex operator+(const Complex& o) const;
  Complex operator-(const Complex& o) const;
  Complex operator*(const Complex& o) const;
  Complex operator/(const Complex& o);
  Complex operator+=(const Complex& o);
  Complex operator-=(const Complex& o);
  Complex operator*=(const Complex& o);
  Complex operator/=(const Complex& o);

  Complex operator+(const T& o);
  Complex operator-(const T& o);
  Complex operator*(const T& o);
  Complex operator/(const T& o);
  Complex operator+=(const T& o);
  Complex operator-=(const T& o);
  Complex operator*=(const T& other);
  Complex operator/=(const T& other);
};

// using polar form z^p = r^p*exp{i*arctan(im/re)*p} == r^p * (cos + i sin)
template<typename T>
inline Complex<T> Complex<T>::pow(const double& p) {
  double r = abs();
  
  if (re == 0) return Complex<T>(T(0), T(std::pow(r, p)));
  if (im == 0) return Complex<T>(T(std::pow(r, p)), T(0));

  double rp = std::pow(r, p);
  double theta = atan(im / re) * p;
  return Complex<T>(T(rp * cos(theta)), T(rp * sin(theta)));
};

template<typename T> 
inline Complex<T> Complex<T>::operator+(const Complex<T>& o) const {
  return Complex<T>(re + o.re, im + o.im);
}

template<typename T>
inline Complex<T> Complex<T>::operator-(const Complex<T>& o) const {
  return Complex<T>(re - o.re, im - o.im);
}

template<typename T>
inline Complex<T> Complex<T>::operator*(const Complex<T>& o) const {
  return Complex<T>(re*o.re-im*o.im, re*o.im + im*o.re);
}

template<typename T>
inline Complex<T> Complex<T>::operator/(const Complex<T>& o) {
  Complex<T> c(o);
  T n = c.abs2();
  return Complex<T>(T((re*o.re+im*o.im)/n), T((im*o.re-re*o.im)/n));
}

template<typename T>
inline Complex<T> Complex<T>::operator+=(const Complex& other) {
  *(this) = (*this) + other;
  return *(this);
}

template<typename T>
inline Complex<T> Complex<T>::operator-=(const Complex& other) {
  *(this) = (*this) - other;
  return *(this);
}

template<typename T>
inline Complex<T> Complex<T>::operator*=(const Complex& other) {
  *(this) = (*this) * other;
  return *(this);
}

template<typename T>
inline Complex<T> Complex<T>::operator/=(const Complex& other) {
  *(this) = (*this) / other;
  return *(this);
}

template<typename T>
inline Complex<T> Complex<T>::operator/=(const T& other) {
  this->real = T(this->real / T(other));
  this->imag = T(this->imag / T(other));
  return *(this);
}

template<typename T>
inline Complex<T> Complex<T>::operator+(const T& other) {
  return Complex<T>(re + other, im);
}

template<typename T>
inline Complex<T> Complex<T>::operator-(const T& other) {
  return Complex<T>(re - other, im);
}

template<typename T>
inline Complex<T> Complex<T>::operator*(const T& other) {
  return Complex<T>(re*other, im*other);
}

template<typename T>
inline Complex<T> Complex<T>::operator/(const T& other) {
  return Complex<T>(re/other, im/other);
}

template<typename T>
inline Complex<T> Complex<T>::operator+=(const T& other) {
  re += other;
  return *(this);
}

template<typename T>
inline Complex<T> Complex<T>::operator-=(const T& other) {
  re -= other;
  return *(this);
}

template<typename T>
inline Complex<T> Complex<T>::operator*=(const T& other) {
  re *= other; im *= other;
  return *(this);
}

template<typename T>
inline Complex<T> operator/(const T& x, const Complex<T>& other) {
  return other.conj()*x / other.abs2();
}

template<typename T>
inline Complex<T> operator*(const T& x, const Complex<T>& other) {
  return other * x;
}

template<typename T>
inline Complex<T> operator+(const T& x, const Complex<T>& other) {
  return other + x;
}

template<typename T>
inline Complex<T> operator-(const T& x, const Complex<T>& other) {
  return other - x;
}

typedef Complex<double> Complex_d;
typedef Complex<float> Complex_f;
typedef Complex<int> Complex_i;
typedef Complex<char> Complex_c;
#endif
