#pragma once
#ifndef MATHLIB_EXPORT_H
#define MATHLIB_EXPORT_H

#include "matrix.h"
#include "vector.h"
#include "vegas.h"

#if defined(_MSC_VER)
  #define EXPORT __declspec(dllexport)
  #define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
  #define EXPORT __attribute__((visibility("default")))
  #define IMPORT
#else
  #define EXPORT
  #define IMPORT
#pragma warning Unknown dynamic link import/export semantics.
#endif

/*vector operations*/
void EXPORT dot_cpu(const char * a, const char * b, char * c, const int dim) {
  Math::Vector::dot_cpu<char>(a, b, c, dim);
}
void EXPORT dot_cpu(const int * a, const int * b, int * c, const int dim) {
  Math::Vector::dot_cpu<int>(a, b, c, dim);
}
void EXPORT dot_cpu(const float * a, const float * b, float * c, const int dim) {
  Math::Vector::dot_cpu<float>(a, b, c, dim);
}
void EXPORT dot_cpu(const double * a, const double * b, double * c, const int dim) {
  Math::Vector::dot_cpu<double>(a, b, c, dim);
}
void EXPORT dot_cpup(const char * a, const char * b, char * c, const int dim) {
  Math::Vector::dot_cpup<char>(a, b, c, dim);
}
void EXPORT dot_cpup(const int * a, const int * b, int * c, const int dim) {
  Math::Vector::dot_cpup<int>(a, b, c, dim);
}
void EXPORT dot_cpup(const float * a, const float * b, float * c, const int dim) {
  Math::Vector::dot_cpup<float>(a, b, c, dim);
}
void EXPORT dot_cpup(const double * a, const double * b, double * c, const int dim) {
  Math::Vector::dot_cpup<double>(a, b, c, dim);
}
//void EXPORT dot_gpu(const char * a, const char * b, char * c, const int dim) {
//  Math::Vector::dot_gpu<char>(a, b, c, dim);
//}
void EXPORT dot_gpu(const int * a, const int * b, int * c, const int dim) {
  Math::Vector::dot_gpu<int>(a, b, c, dim);
}
void EXPORT dot_gpu(const float * a, const float * b, float * c, const int dim) {
  Math::Vector::dot_gpu<float>(a, b, c, dim);
}
//void EXPORT dot_gpu(const double * a, const double * b, double * c, const int dim) {
//  Math::Vector::dot_gpu<double>(a, b, c, dim);
//}

/*matrix multiplication*/
void EXPORT mm_cpu(const char * a, const char * b, char * c, const int ra, const int ca, const int cb) {
  Math::Matrix::axb_cpu<char>(a, b, c, ra, ca, cb);
}
void EXPORT mm_cpu(const int * a, const int * b, int * c, const int ra, const int ca, const int cb) {
  Math::Matrix::axb_cpu<int>(a, b, c, ra, ca, cb);
}
void EXPORT mm_cpu(const float * a, const float * b, float * c, const int ra, const int ca, const int cb) {
  Math::Matrix::axb_cpu<float>(a, b, c, ra, ca, cb);
}
void EXPORT mm_cpu(const double * a, const double * b, double * c, const int ra, const int ca, const int cb) {
  Math::Matrix::axb_cpu<double>(a, b, c, ra, ca, cb);
}
void EXPORT mm_cpup(const char * a, const char * b, char * c, const int ra, const int ca, const int cb) {
  Math::Matrix::axb_cpup<char>(a, b, c, ra, ca, cb);
}
void EXPORT mm_cpup(const int * a, const int * b, int * c, const int ra, const int ca, const int cb) {
  Math::Matrix::axb_cpup<int>(a, b, c, ra, ca, cb);
}
void EXPORT mm_cpup(const float * a, const float * b, float * c, const int ra, const int ca, const int cb) {
  Math::Matrix::axb_cpup<float>(a, b, c, ra, ca, cb);
}
void EXPORT mm_cpup(const double * a, const double * b, double * c, const int ra, const int ca, const int cb) {
  Math::Matrix::axb_cpup<double>(a, b, c, ra, ca, cb);
}
void EXPORT mm_gpu(const char * a, const char * b, char * c, const int ra, const int ca, const int cb) {
  Math::Matrix::axb_gpu<char>(a, b, c, ra, ca, cb);
}
void EXPORT mm_gpu(const int * a, const int * b, int * c, const int ra, const int ca, const int cb) {
  Math::Matrix::axb_gpu<int>(a, b, c, ra, ca, cb);
}
void EXPORT mm_gpu(const float * a, const float * b, float * c, const int ra, const int ca, const int cb) {
  Math::Matrix::axb_gpu<float>(a, b, c, ra, ca, cb);
}
void EXPORT mm_gpu(const double * a, const double * b, double * c, const int ra, const int ca, const int cb) {
  Math::Matrix::axb_gpu<double>(a, b, c, ra, ca, cb);
}

/*LU-decomposition*/
void EXPORT lu_cpu(char * a, char * b, int * p, const int N) {
  Math::Matrix::lud_cpu<char>(a, b, p, N);
}
void EXPORT lu_cpu(int * a, int * b, int * p, const int N) {
  Math::Matrix::lud_cpu<int>(a, b, p, N);
}
void EXPORT lu_cpu(float * a, float * b, int * p, const int N) {
  Math::Matrix::lud_cpu<float>(a, b, p, N);
}
void EXPORT lu_cpu(double * a, double * b, int * p, const int N) {
  Math::Matrix::lud_cpu<double>(a, b, p, N);
}
void EXPORT lu_cpup(char * a, char * b, int * p, const int N) {
  Math::Matrix::lud_cpup<char>(a, b, p, N);
}
void EXPORT lu_cpup(int * a, int * b, int * p, const int N) {
  Math::Matrix::lud_cpup<int>(a, b, p, N);
}
void EXPORT lu_cpup(float * a, float * b, int * p, const int N) {
  Math::Matrix::lud_cpup<float>(a, b, p, N);
}
void EXPORT lu_cpup(double * a, double * b, int * p, const int N) {
  Math::Matrix::lud_cpup<double>(a, b, p, N);
}
void EXPORT lu_gpu(char * a, char * b, int * p, const int N) {
  Math::Matrix::lud_gpu<char>(a, b, p, N);
}
void EXPORT lu_gpu(int * a, int * b, int * p, const int N) {
  Math::Matrix::lud_gpu<int>(a, b, p, N);
}
void EXPORT lu_gpu(float * a, float * b, int * p, const int N) {
  Math::Matrix::lud_gpu<float>(a, b, p, N);
}
void EXPORT lu_gpu(double * a, double * b, int * p, const int N) {
  Math::Matrix::lud_gpu<double>(a, b, p, N);
}

/*QR-decomposition*/
void EXPORT qr_cpu(const char * a, char * Q, char * R, const int N) {
  Math::Matrix::qr_cpu<char>(a, Q, R, N);
}
void EXPORT qr_cpu(const int * a, int * Q, int * R, const int N) {
  Math::Matrix::qr_cpu<int>(a, Q, R, N);
}
void EXPORT qr_cpu(const float * a, float * Q, float * R, const int N) {
  Math::Matrix::qr_cpu<float>(a, Q, R, N);
}
void EXPORT qr_cpu(const double * a, double * Q, double * R, const int N) {
  Math::Matrix::qr_cpu<double>(a, Q, R, N);
}

/*Gaussian Elimination*/
void EXPORT ge_cpu(char * a, char * b, const int N) {
  Math::Matrix::ge_cpu<char>(a, b, N);
}
void EXPORT ge_cpu(int * a, int * b, const int N) {
  Math::Matrix::ge_cpu<int>(a, b, N);
}
void EXPORT ge_cpu(float * a, float * b, const int N) {
  Math::Matrix::ge_cpu<float>(a, b, N);
}
void EXPORT ge_cpu(double * a, double * b, const int N) {
  Math::Matrix::ge_cpu<double>(a, b, N);
}
void EXPORT ge_cpu_tri(char * a, char * b, const int N) {
  Math::Matrix::ge_cpu_tri<char>(a, b, N);
}
void EXPORT ge_cpu_tri(int * a, int * b, const int N) {
  Math::Matrix::ge_cpu_tri<int>(a, b, N);
}
void EXPORT ge_cpu_tri(float * a, float * b, const int N) {
  Math::Matrix::ge_cpu_tri<float>(a, b, N);
}
void EXPORT ge_cpu_tri(double * a, double * b, const int N) {
  Math::Matrix::ge_cpu_tri<double>(a, b, N);
}

/*Integration*/
void EXPORT vegas_cpu(vegas_integrand f, void * params, double xl[], double xu[], size_t dim, size_t icalls, double * res, double * abserr) {
  Math::Integrate::Vegas(f, params, xl, xu, dim, icalls, res, abserr);
}
void EXPORT vegas_cpup(vegas_integrand f, void * params, double xl[], double xu[], size_t dim, size_t icalls, double * res, double * abserr) {
  Math::Integrate::Vegasp(f, params, xl, xu, dim, icalls, res, abserr);
}
#endif
