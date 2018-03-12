//#pragma once

//#ifndef MATHLIB_MATRIX2_H
//#define MATHLIB_MATRIX2_H

#include <stdio.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <random>

#include "matrix.h"
#include "clock.h"

template<typename T>
void naive_mm(const std::unique_ptr<T>& A,
	      const std::unique_ptr<T>& B,
	      std::unique_ptr<T>& C,
	      const unsigned int rows,
	      const unsigned int cols,
	      const unsigned int bsize);

template<typename T>
void sub_matrix(const std::unique_ptr<T>& src,
		std::unique_ptr<T>& dst,
		const unsigned int sidx,
		const unsigned int srow,
		const unsigned int scol,
		const unsigned int max_row,
		const unsigned int max_col,
		const unsigned int rows,
		const unsigned int cols);

template<typename T> inline void zero(std::unique_ptr<T>& v, const unsigned int N);

template<typename T>
void mm(const std::unique_ptr<T>& A,
	const std::unique_ptr<T>& B,
	std::unique_ptr<T>& C,
	const size_t rA,
	const size_t cA,
	const size_t cB);

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
typename std::enable_if
<
    std::is_array<T>::value,
    std::unique_ptr<T>
>::type
make_unique(std::size_t n) {
    typedef typename std::remove_extent<T>::type RT;
    return std::unique_ptr<T>(new RT[n]);
}

template<typename T>
inline void mm_check(const std::unique_ptr<T>& A,
		     const std::unique_ptr<T>& B,
		     std::unique_ptr<T>& C,
		     const size_t rA,
		     const size_t cA,
		     const size_t cB) {
  for (unsigned int r=0, idx=0; r<rA; ++r) {
    for (unsigned int c=0; c<cB; ++c, ++idx) {
      const int s = r*cA;
      for (unsigned int j=0; j<cA; ++j) C[idx] += A[s+j] * B[j*cB + c];       
    }
  }
}

inline void print_matrix(float M[], const char * msg) {  
  printf("......%s......\n", msg);
  for (unsigned int r=0,idx=0; r<16; ++r) {
    for (unsigned int c=0; c<16; ++c,++idx) {
      printf("%3.3f ", M[idx]);
    }
    printf("\n");
  }
}

template<typename T>
inline void print_matrix2(const std::unique_ptr<T>& M, const char * msg) {  
  printf("......%s......\n", msg);
  for (unsigned int r=0,idx=0; r<32; ++r) {
    for (unsigned int c=0; c<32; ++c,++idx) {
      printf("%3.3f ", M[idx]);
    }
    printf("\n");
  }
}

//////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  
  const size_t N_ra = 2263;
  const size_t N_ca = 2257;
  const size_t N_rb = N_ca;
  const size_t N_cb = 2231;
  
  const size_t N_rc = N_ra;
  const size_t N_cc = N_cb;
    
  auto A = make_unique<float[]>(N_ra*N_ca);
  auto B = make_unique<float[]>(N_rb*N_cb);
  auto C = make_unique<float[]>(N_rc*N_cc);
  auto D = make_unique<float[]>(N_rc*N_cc);


  for (auto j=0; j<N_ra*N_ca; ++j) A[j] = j;
  for (auto j=0; j<N_rb*N_cb; ++j) B[j] = j;
  for (auto j=0; j<N_rc*N_cc; ++j) { C[j] = 0; D[j] = 0; }
  
  
  Clock clock("mm");
  clock.start();
  mm(A, B, C, N_ra, N_ca, N_cb);
  clock.finished();
  printf("..finished fast (new)\n");

  //clock.start();
  //Math::Matrix::axb_cpu(A.get(), B.get(), D.get(), N, N, N);
  //printf("..finished fast (old)\n");
  //clock.finished();

  clock.start();
  mm_check(A, B, D, N_ra, N_ca, N_cb);
  clock.finished();
  printf("..finished slow\n");
  
  size_t nerrs = 0;
  for (unsigned int r=0,idx=0; r<N_rc; ++r){
    for (unsigned int c=0; c<N_cc; ++c, ++idx) {
      if (fabs(C[idx]-D[idx]) >= 1e-4) ++nerrs;
    }
  }
  printf("..finished with %lu-errors\n", nerrs);  
  
  return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void naive_mm(const std::unique_ptr<T>& A,
	      const std::unique_ptr<T>& B,
	      std::unique_ptr<T>& C,
	      const unsigned int rows,
	      const unsigned int cols,
	      const unsigned int bsize) {
  for (auto r=0; r<rows; ++r) {
    for (auto c=0; c<cols; ++c) {
      auto s = bsize * r;
      for (auto j=0; j<bsize; ++j) C[s+c] += A[s+j]*B[bsize*j+c];
    }
  }
}

template<typename T>
void sub_matrix(const std::unique_ptr<T>& src,
		std::unique_ptr<T>& dst,
		const unsigned int sidx,
		const unsigned int srow,
		const unsigned int scol,
		const unsigned int max_row,
		const unsigned int max_col,
		const unsigned int rows,
		const unsigned int cols) {
  zero(dst, rows*cols);  
  
  for (unsigned int r = 0, dst_idx = 0; r < rows; ++r) {
    for (unsigned int c = 0; c < cols; ++c, ++dst_idx) {
      if (srow + r >= max_row || scol + c >= max_col) continue;
      unsigned int src_idx = sidx + (r*max_col + c);
      dst[dst_idx] = src[src_idx];
    }
  }
}

template<typename T>
inline void zero(std::unique_ptr<T>& v, const unsigned int N) {
  for (unsigned int j=0; j<N; ++j) v[j] = 0;
}

template<typename T>
void mm(const std::unique_ptr<T>& A,
	const std::unique_ptr<T>& B,
	std::unique_ptr<T>& C,
	const size_t rA,
	const size_t cA,
	const size_t cB) {

  const size_t bsize = 64;
  
  // smaller matrices are handled using naive matrix multiplication  
  if (rA < 3*bsize &&
      cA < 3*bsize &&
      cB < 3*bsize) {
    mm_check(A, B, C, rA, cA, cB);
    return;
  }
  
  
  auto sa = make_unique<float[]>(bsize*bsize); // submatrix storing blocks of A
  auto sb = make_unique<float[]>(bsize*bsize); // submatrix storing blocks of B
  auto sc = make_unique<float[]>(bsize*bsize); // submatrix storing the result sa*sb = sc (copied to C)
  
  auto col_blocks = static_cast<unsigned int>(std::ceil((double)cB/(double)bsize));
  auto row_blocks = static_cast<unsigned int>(std::ceil((double)rA/(double)bsize));
  auto inner_mults = static_cast<unsigned int>(std::ceil((double)cA/(double)bsize));
  
  for (unsigned int ridx = 0, rmx = row_blocks*bsize; ridx < rmx; ridx += bsize) {
    for (unsigned int cidx = 0, cmx = col_blocks*bsize; cidx < cmx; cidx += bsize) {      
      unsigned int astart = ridx * cA; // bstart = cidx
      unsigned int bstart = cidx;
      zero(sc, bsize*bsize);
      
      for (unsigned int iidx = 0, imx = inner_mults*bsize; iidx < imx; iidx += bsize) {
	// move right by bsize x bsize blocks in A, copy them to sa
	sub_matrix(A, sa, astart + iidx, ridx, iidx, rA, cA, bsize, bsize);
	// move down in bsize x bsize blocks in B copy them to sb
	sub_matrix(B, sb, bstart + cB*iidx, iidx, cidx, cA, cB, bsize, bsize);

	auto sa_rows = std::min(rA - ridx, bsize);
	auto sb_cols = std::min(cB - cidx, bsize);	
	naive_mm(sa, sb, sc, sa_rows, sb_cols, bsize); // multiply sc = sa*sb, store results to sc
      }
      
      // copy result to C
      for (unsigned int r=ridx, idx=0; r<ridx+bsize; ++r) {
	for (unsigned int c=cidx; c<cidx+bsize; ++c,++idx) {
	  if (r >= rA || c >= cB) continue;
	  auto cc = r * cB + c;
	  C[cc] = sc[idx];
	}
      }      
    }
  }    
}
