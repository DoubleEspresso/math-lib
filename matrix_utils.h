#pragma once

#ifndef MATHLIB_MATRIX_UTILS_H
#define MATHLIB_MATRIX_UTILS_H

#include <iostream>
#include <limits>
#define INF std::numeric_limits<double>::infinity()

// note : cpu matrix utility class containing some hard-coded
// matrix multiplication and specialized strassen multiplication 
// for smaller matrices (to be used to speed up matrix multiplication on the cpu)
template<typename T>
inline void mm8x8_fast(const T * a, const T * b, T * c) {
  c[0] = a[0] * b[0] + a[1] * b[8] + a[2] * b[16] + a[3] * b[24] + a[4] * b[32] + a[5] * b[40] + a[6] * b[48] + a[7] * b[56];
  c[1] = a[0] * b[1] + a[1] * b[9] + a[2] * b[17] + a[3] * b[25] + a[4] * b[33] + a[5] * b[41] + a[6] * b[49] + a[7] * b[57];
  c[2] = a[0] * b[2] + a[1] * b[10] + a[2] * b[18] + a[3] * b[26] + a[4] * b[34] + a[5] * b[42] + a[6] * b[50] + a[7] * b[58];
  c[3] = a[0] * b[3] + a[1] * b[11] + a[2] * b[19] + a[3] * b[27] + a[4] * b[35] + a[5] * b[43] + a[6] * b[51] + a[7] * b[59];
  c[4] = a[0] * b[4] + a[1] * b[12] + a[2] * b[20] + a[3] * b[28] + a[4] * b[36] + a[5] * b[44] + a[6] * b[52] + a[7] * b[60];
  c[5] = a[0] * b[5] + a[1] * b[13] + a[2] * b[21] + a[3] * b[29] + a[4] * b[37] + a[5] * b[45] + a[6] * b[53] + a[7] * b[61];
  c[6] = a[0] * b[6] + a[1] * b[14] + a[2] * b[22] + a[3] * b[30] + a[4] * b[38] + a[5] * b[46] + a[6] * b[54] + a[7] * b[62];
  c[7] = a[0] * b[7] + a[1] * b[15] + a[2] * b[23] + a[3] * b[31] + a[4] * b[39] + a[5] * b[47] + a[6] * b[55] + a[7] * b[63];

  c[8] = a[8] * b[0] + a[9] * b[8] + a[10] * b[16] + a[11] * b[24] + a[12] * b[32] + a[13] * b[40] + a[14] * b[48] + a[15] * b[56];
  c[9] = a[8] * b[1] + a[9] * b[9] + a[10] * b[17] + a[11] * b[25] + a[12] * b[33] + a[13] * b[41] + a[14] * b[49] + a[15] * b[57];
  c[10] = a[8] * b[2] + a[9] * b[10] + a[10] * b[18] + a[11] * b[26] + a[12] * b[34] + a[13] * b[42] + a[14] * b[50] + a[15] * b[58];
  c[11] = a[8] * b[3] + a[9] * b[11] + a[10] * b[19] + a[11] * b[27] + a[12] * b[35] + a[13] * b[43] + a[14] * b[51] + a[15] * b[59];
  c[12] = a[8] * b[4] + a[9] * b[12] + a[10] * b[20] + a[11] * b[28] + a[12] * b[36] + a[13] * b[44] + a[14] * b[52] + a[15] * b[60];
  c[13] = a[8] * b[5] + a[9] * b[13] + a[10] * b[21] + a[11] * b[29] + a[12] * b[37] + a[13] * b[45] + a[14] * b[53] + a[15] * b[61];
  c[14] = a[8] * b[6] + a[9] * b[14] + a[10] * b[22] + a[11] * b[30] + a[12] * b[38] + a[13] * b[46] + a[14] * b[54] + a[15] * b[62];
  c[15] = a[8] * b[7] + a[9] * b[15] + a[10] * b[23] + a[11] * b[31] + a[12] * b[39] + a[13] * b[47] + a[14] * b[55] + a[15] * b[63];

  c[16] = a[16] * b[0] + a[17] * b[8] + a[18] * b[16] + a[19] * b[24] + a[20] * b[32] + a[21] * b[40] + a[22] * b[48] + a[23] * b[56];
  c[17] = a[16] * b[1] + a[17] * b[9] + a[18] * b[17] + a[19] * b[25] + a[20] * b[33] + a[21] * b[41] + a[22] * b[49] + a[23] * b[57];
  c[18] = a[16] * b[2] + a[17] * b[10] + a[18] * b[18] + a[19] * b[26] + a[20] * b[34] + a[21] * b[42] + a[22] * b[50] + a[23] * b[58];
  c[19] = a[16] * b[3] + a[17] * b[11] + a[18] * b[19] + a[19] * b[27] + a[20] * b[35] + a[21] * b[43] + a[22] * b[51] + a[23] * b[59];
  c[20] = a[16] * b[4] + a[17] * b[12] + a[18] * b[20] + a[19] * b[28] + a[20] * b[36] + a[21] * b[44] + a[22] * b[52] + a[23] * b[60];
  c[21] = a[16] * b[5] + a[17] * b[13] + a[18] * b[21] + a[19] * b[29] + a[20] * b[37] + a[21] * b[45] + a[22] * b[53] + a[23] * b[61];
  c[22] = a[16] * b[6] + a[17] * b[14] + a[18] * b[22] + a[19] * b[30] + a[20] * b[38] + a[21] * b[46] + a[22] * b[54] + a[23] * b[62];
  c[23] = a[16] * b[7] + a[17] * b[15] + a[18] * b[23] + a[19] * b[31] + a[20] * b[39] + a[21] * b[47] + a[22] * b[55] + a[23] * b[63];

  c[24] = a[24] * b[0] + a[25] * b[8] + a[26] * b[16] + a[27] * b[24] + a[28] * b[32] + a[29] * b[40] + a[30] * b[48] + a[31] * b[56];
  c[25] = a[24] * b[1] + a[25] * b[9] + a[26] * b[17] + a[27] * b[25] + a[28] * b[33] + a[29] * b[41] + a[30] * b[49] + a[31] * b[57];
  c[26] = a[24] * b[2] + a[25] * b[10] + a[26] * b[18] + a[27] * b[26] + a[28] * b[34] + a[29] * b[42] + a[30] * b[50] + a[31] * b[58];
  c[27] = a[24] * b[3] + a[25] * b[11] + a[26] * b[19] + a[27] * b[27] + a[28] * b[35] + a[29] * b[43] + a[30] * b[51] + a[31] * b[59];
  c[28] = a[24] * b[4] + a[25] * b[12] + a[26] * b[20] + a[27] * b[28] + a[28] * b[36] + a[29] * b[44] + a[30] * b[52] + a[31] * b[60];
  c[29] = a[24] * b[5] + a[25] * b[13] + a[26] * b[21] + a[27] * b[29] + a[28] * b[37] + a[29] * b[45] + a[30] * b[53] + a[31] * b[61];
  c[30] = a[24] * b[6] + a[25] * b[14] + a[26] * b[22] + a[27] * b[30] + a[28] * b[38] + a[29] * b[46] + a[30] * b[54] + a[31] * b[62];
  c[31] = a[24] * b[7] + a[25] * b[15] + a[26] * b[23] + a[27] * b[31] + a[28] * b[39] + a[29] * b[47] + a[30] * b[55] + a[31] * b[63];

  c[32] = a[32] * b[0] + a[33] * b[8] + a[34] * b[16] + a[35] * b[24] + a[36] * b[32] + a[37] * b[40] + a[38] * b[48] + a[39] * b[56];
  c[33] = a[32] * b[1] + a[33] * b[9] + a[34] * b[17] + a[35] * b[25] + a[36] * b[33] + a[37] * b[41] + a[38] * b[49] + a[39] * b[57];
  c[34] = a[32] * b[2] + a[33] * b[10] + a[34] * b[18] + a[35] * b[26] + a[36] * b[34] + a[37] * b[42] + a[38] * b[50] + a[39] * b[58];
  c[35] = a[32] * b[3] + a[33] * b[11] + a[34] * b[19] + a[35] * b[27] + a[36] * b[35] + a[37] * b[43] + a[38] * b[51] + a[39] * b[59];
  c[36] = a[32] * b[4] + a[33] * b[12] + a[34] * b[20] + a[35] * b[28] + a[36] * b[36] + a[37] * b[44] + a[38] * b[52] + a[39] * b[60];
  c[37] = a[32] * b[5] + a[33] * b[13] + a[34] * b[21] + a[35] * b[29] + a[36] * b[37] + a[37] * b[45] + a[38] * b[53] + a[39] * b[61];
  c[38] = a[32] * b[6] + a[33] * b[14] + a[34] * b[22] + a[35] * b[30] + a[36] * b[38] + a[37] * b[46] + a[38] * b[54] + a[39] * b[62];
  c[39] = a[32] * b[7] + a[33] * b[15] + a[34] * b[23] + a[35] * b[31] + a[36] * b[39] + a[37] * b[47] + a[38] * b[55] + a[39] * b[63];

  c[40] = a[40] * b[0] + a[41] * b[8] + a[42] * b[16] + a[43] * b[24] + a[44] * b[32] + a[45] * b[40] + a[46] * b[48] + a[47] * b[56];
  c[41] = a[40] * b[1] + a[41] * b[9] + a[42] * b[17] + a[43] * b[25] + a[44] * b[33] + a[45] * b[41] + a[46] * b[49] + a[47] * b[57];
  c[42] = a[40] * b[2] + a[41] * b[10] + a[42] * b[18] + a[43] * b[26] + a[44] * b[34] + a[45] * b[42] + a[46] * b[50] + a[47] * b[58];
  c[43] = a[40] * b[3] + a[41] * b[11] + a[42] * b[19] + a[43] * b[27] + a[44] * b[35] + a[45] * b[43] + a[46] * b[51] + a[47] * b[59];
  c[44] = a[40] * b[4] + a[41] * b[12] + a[42] * b[20] + a[43] * b[28] + a[44] * b[36] + a[45] * b[44] + a[46] * b[52] + a[47] * b[60];
  c[45] = a[40] * b[5] + a[41] * b[13] + a[42] * b[21] + a[43] * b[29] + a[44] * b[37] + a[45] * b[45] + a[46] * b[53] + a[47] * b[61];
  c[46] = a[40] * b[6] + a[41] * b[14] + a[42] * b[22] + a[43] * b[30] + a[44] * b[38] + a[45] * b[46] + a[46] * b[54] + a[47] * b[62];
  c[47] = a[40] * b[7] + a[41] * b[15] + a[42] * b[23] + a[43] * b[31] + a[44] * b[39] + a[45] * b[47] + a[46] * b[55] + a[47] * b[63];

  c[48] = a[48] * b[0] + a[49] * b[8] + a[50] * b[16] + a[51] * b[24] + a[52] * b[32] + a[53] * b[40] + a[54] * b[48] + a[55] * b[56];
  c[49] = a[48] * b[1] + a[49] * b[9] + a[50] * b[17] + a[51] * b[25] + a[52] * b[33] + a[53] * b[41] + a[54] * b[49] + a[55] * b[57];
  c[50] = a[48] * b[2] + a[49] * b[10] + a[50] * b[18] + a[51] * b[26] + a[52] * b[34] + a[53] * b[42] + a[54] * b[50] + a[55] * b[58];
  c[51] = a[48] * b[3] + a[49] * b[11] + a[50] * b[19] + a[51] * b[27] + a[52] * b[35] + a[53] * b[43] + a[54] * b[51] + a[55] * b[59];
  c[52] = a[48] * b[4] + a[49] * b[12] + a[50] * b[20] + a[51] * b[28] + a[52] * b[36] + a[53] * b[44] + a[54] * b[52] + a[55] * b[60];
  c[53] = a[48] * b[5] + a[49] * b[13] + a[50] * b[21] + a[51] * b[29] + a[52] * b[37] + a[53] * b[45] + a[54] * b[53] + a[55] * b[61];
  c[54] = a[48] * b[6] + a[49] * b[14] + a[50] * b[22] + a[51] * b[30] + a[52] * b[38] + a[53] * b[46] + a[54] * b[54] + a[55] * b[62];
  c[55] = a[48] * b[7] + a[49] * b[15] + a[50] * b[23] + a[51] * b[31] + a[52] * b[39] + a[53] * b[47] + a[54] * b[55] + a[55] * b[63];

  c[56] = a[56] * b[0] + a[57] * b[8] + a[58] * b[16] + a[59] * b[24] + a[60] * b[32] + a[61] * b[40] + a[62] * b[48] + a[63] * b[56];
  c[57] = a[56] * b[1] + a[57] * b[9] + a[58] * b[17] + a[59] * b[25] + a[60] * b[33] + a[61] * b[41] + a[62] * b[49] + a[63] * b[57];
  c[58] = a[56] * b[2] + a[57] * b[10] + a[58] * b[18] + a[59] * b[26] + a[60] * b[34] + a[61] * b[42] + a[62] * b[50] + a[63] * b[58];
  c[59] = a[56] * b[3] + a[57] * b[11] + a[58] * b[19] + a[59] * b[27] + a[60] * b[35] + a[61] * b[43] + a[62] * b[51] + a[63] * b[59];
  c[60] = a[56] * b[4] + a[57] * b[12] + a[58] * b[20] + a[59] * b[28] + a[60] * b[36] + a[61] * b[44] + a[62] * b[52] + a[63] * b[60];
  c[61] = a[56] * b[5] + a[57] * b[13] + a[58] * b[21] + a[59] * b[29] + a[60] * b[37] + a[61] * b[45] + a[62] * b[53] + a[63] * b[61];
  c[62] = a[56] * b[6] + a[57] * b[14] + a[58] * b[22] + a[59] * b[30] + a[60] * b[38] + a[61] * b[46] + a[62] * b[54] + a[63] * b[62];
  c[63] = a[56] * b[7] + a[57] * b[15] + a[58] * b[23] + a[59] * b[31] + a[60] * b[39] + a[61] * b[47] + a[62] * b[55] + a[63] * b[63];
}

template<typename T>
inline void submat(const T * a, const T *& b, const int s) {
  b = &a[s];
}

template<typename T>
inline void submat(const T * a, T * b, const int sa, const int stride, const int row, const int col) {
  // note : returns a square matrix of size NxN
  int ss = sa;
  for (int r = 0, i = 0; r < row; ++r) {
    for (int c = 0; c < col; ++c, ++i) {
      b[i] = a[ss + c];
    }
    ss += stride;
  }
}

template<typename T>
inline void subcol(const T * a, T * b, const int sa, const int stride, const int N) {
  for (int j = sa, idx = 0; idx < N; j += stride, ++idx) b[idx] = a[j];
}

template<typename T>
inline void subrow(const T * a, T * b, const int sa, const int N) {
  for (int j = sa, idx = 0; idx < N; ++j, ++idx) b[idx] = a[j];
}

template<typename T>
inline void add8x8(const T * a, const T * b, T * c) {
  for (int j = 0; j < 64; ++j) c[j] = a[j] + b[j];
}

template<typename T>
inline void sub8x8(const T * a, const T * b, T * c) {
  for (int j = 0; j < 64; ++j) c[j] = a[j] - b[j];
}

template<typename T>
inline void swap(T * a, const int ra, const int ca, const int i1, const int i2, const int nrows, const int ncols) {
  // purpose: in place swap of a "chunk" of matrix "a".
  // ra: rows of a
  // ca: cols of a
  // i1: start index in row-major matrix a
  // i2: 2nd start index (to be replaced)
  // nrows: nb rows to swap
  // ncols: nb cols to swap
  if (nrows > ra) return;

  T tmp = T(0);
  for (int r = 0; r < nrows; ++r) {
    for (int c = 0; c < ncols; ++c) {
      int aidx = r * ca + c;
      tmp = a[i2 + aidx];
      a[i2 + aidx] = a[i1 + aidx];
      a[i1 + aidx] = tmp;
    }
  }
}

// finds most non-zero element along a column
// accounting for possible permutations of rows and examining
// only those rows > srow
template<typename T>
inline void mostnonzero_col(const T * a, int * P, const int srow, const int scol, T * mx, int * idx, const int nrows, const int ncols) {
  *mx = -T(INF); *idx = srow;
  for (int r = srow; r < nrows; ++r) {
    T t = a[P[r] * ncols + scol];
    t = (t < 0 ? -t : t);
    if (t >(*mx)) { (*mx) = t; (*idx) = r; }
  }
}

template<typename T>
inline void swap_rows(T * a, const int ra, const int ca, const int i1, const int i2) {
  swap(a, ra, ca, i1, i2, 1, ca);
}

template<typename T>
inline void max_col(const T * a, const int sidx, T * mx, int * idx, const int nrows, const int ncols) {
  *mx = -T(INF); *idx = 0; int srow = sidx / nrows;
  for (int c = sidx, max = nrows * ncols, i = 0; c < max; c += ncols, ++i) {
    if (a[c] > (*mx)) { (*mx) = a[c]; (*idx) = srow + i; }
  }
}

template<typename T>
inline void strassen16x16(const T * a, const T * b, T * c) {
  T a11[64]; submat<T>(a, a11, 0, 16, 8, 8);
  T a12[64]; submat<T>(a, a12, 8, 16, 8, 8);
  T a21[64]; submat<T>(a, a21, 128, 16, 8, 8);
  T a22[64]; submat<T>(a, a22, 136, 16, 8, 8);

  T b11[64]; submat<T>(b, b11, 0, 16, 8, 8);
  T b12[64]; submat<T>(b, b12, 8, 16, 8, 8);
  T b21[64]; submat<T>(b, b21, 128, 16, 8, 8);
  T b22[64]; submat<T>(b, b22, 136, 16, 8, 8);

  /* additions */
  T mm1[64]; add8x8<T>(a11, a22, mm1);
  T mm2[64]; add8x8<T>(b11, b22, mm2);

  T m1[64]; mm8x8_fast<T>(mm1, mm2, m1);

  add8x8<T>(a21, a22, mm1);
  T m2[64]; mm8x8_fast<T>(mm1, b11, m2);

  sub8x8<T>(b12, b22, mm1);
  T m3[64]; mm8x8_fast<T>(a11, mm1, m3);

  sub8x8<T>(b21, b11, mm1);
  T m4[64]; mm8x8_fast<T>(a22, mm1, m4);

  add8x8<T>(a11, a12, mm1);

  T m5[64]; mm8x8_fast<T>(mm1, b22, m5);

  sub8x8<T>(a21, a11, mm1);  add8x8<T>(b11, b12, mm2);
  T m6[64]; mm8x8_fast<T>(mm1, mm2, m6);

  sub8x8(a12, a22, mm1);  add8x8(b21, b22, mm2);
  T m7[64]; mm8x8_fast<T>(mm1, mm2, m7);

  /*compute c-matrices*/
  T c11[64];
  add8x8<T>(m1, m4, mm1); sub8x8<T>(m7, m5, mm2); add8x8<T>(mm1, mm2, c11);

  T c12[64];
  add8x8<T>(m3, m5, c12);

  T c21[64];
  add8x8<T>(m2, m4, c21);

  T c22[64];
  sub8x8<T>(m1, m2, mm1); add8x8<T>(m3, m6, mm2); add8x8<T>(mm1, mm2, c22);

  /* copy & return */
  int i = 0; int ss = 0;
  for (int r = 0; r < 8; ++r) {
    for (int cc = 0; cc < 8; ++cc) {
      c[ss + cc] = c11[i];
      c[ss + cc + 8] = c12[i];

      c[ss + 128 + cc] = c21[i];
      c[ss + 136 + cc] = c22[i++];
    }
    ss += 16;
  }
}
#endif
