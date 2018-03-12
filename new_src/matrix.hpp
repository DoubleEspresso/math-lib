#include "memdetail.h"

namespace Math {
  namespace Matrix {


    // non-blocked (naive) implementation of matrix multiplication
    // meant for small (N~256) sized matrices
    template<typename T>
    inline void axb_naive(const std::unique_ptr<T>& A,
			  const std::unique_ptr<T>& B,
			  std::unique_ptr<T>& C,
			  const size_t rA,
			  const size_t cA,
			  const size_t cB)
    {
      for (unsigned int r=0, idx=0; r<rA; ++r) {
	for (unsigned int c=0; c<cB; ++c, ++idx) {
	  const int s = r*cA;
	  for (unsigned int j=0; j<cA; ++j) C[idx] += A[s+j] * B[j*cB + c];       
	}
      }
    }

    // private helper method to zero an array
    template<typename T>
    inline void zero(std::unique_ptr<T>& v, const unsigned int N) {
      for (auto j=0; j<N; ++j) v[j] = 0;
    }
    
    // private helper method for blocked matrix multiplication
    // purpose is to collect a submatrix from src, and copy it
    // to dst array
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
      
      for (auto r = 0, dst_idx = 0; r < rows; ++r) {
	for (auto c = 0; c < cols; ++c, ++dst_idx) {
	  if (srow + r >= max_row || scol + c >= max_col) continue;
	  auto src_idx = sidx + (r*max_col + c);
	  dst[dst_idx] = src[src_idx];
	}
      }
    }
    
    // single-threaded blocked matrix multiplication on the cpu
    // C = AxB
    template<typename T>
    void axb(const std::unique_ptr<T>& A,
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
	axb_naive(A, B, C, rA, cA, cB);
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
	    
	    // inner matrix multiplication on blocks sc = sa * sb
	    for (auto r=0; r<sa_rows; ++r) {
	      for (auto c=0; c<sb_cols; ++c) {
		auto s = bsize * r;
		for (auto j=0; j<bsize; ++j) sc[s+c] += sa[s+j]*sb[bsize*j+c];
	      }					       
	    }	    
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

    
  };
};
