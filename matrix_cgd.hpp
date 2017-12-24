
//------------------------------------------------------------------
// inverter utilities
//------------------------------------------------------------------
template<typename T> inline bool is_ok(T * x, const int N, double tol) {
  double r = 0;
  for (int j=0; j<N; ++j) r += x[j]*x[j];
  return sqrt(r) <= tol;
  //for (int j=0; j<N; ++j) if (fabs(x[j]) > tol) return false;
  //return true;
}

//------------------------------------------------------------------
// main inverter methods
//  - naive conjugate gradient descent (no-preconditioning)
//  - biconjugate gradient stabilized (BiCStab) (no pre-conditioning)
//------------------------------------------------------------------
namespace Math {
  namespace Matrix {

    // naive conjugate gradient descent (used as baseline comparison)
    template<typename T> void CGD(const T * A, T * X, T * B, const int N) {
      // create a residual vector "r", which is a copy of B
      // this is equivalent to an initial X = 0
      T * r = new T[N]; memset(r, 0, N * sizeof(T));
      T * P = new T[N]; memset(P, 0, N * sizeof(T));
      T * x = new T[N]; memset(x, 0, N * sizeof(T));
      T * S = new T[N]; memset(S, 0, N * sizeof(T)); // storage for A.pk
      T a = 0; 
      T b = 0;
      double tol = 1e-10; size_t maxiter = N*100;
    
      for (int j=0; j<N; ++j) {
	r[j] = B[j]; 
	x[j] = 0;
	P[j] = r[j];
      }
      T rr = 0;
      for (int j=0; j<N; ++j) rr += r[j] * r[j];      
      size_t iter = 0;

      while(true) { 
	// 1. compute alpha coefficient
	//Math::Matrix::DGEMM(A, P, S, N, 1, N); // todo: BLAS option (?)
	//Math::Matrix::axb_cpup(A, P, S, N, N, 1);
	Math::Matrix::axb_cpup(A, P, S, N, N, 1);
	T pAp = 0;
	for (int j=0; j<N; ++j) pAp += P[j]*S[j];
	a = rr / pAp;

	// 2. update x & r
	for (int j=0; j<N; ++j) {
	  x[j] += a * P[j];
	  r[j] -= a * S[j];
	}
	if(is_ok(r, N, tol)) break; 

	// 3. update P
	T rrN = 0; 
	for (int j=0; j<N; ++j) rrN += r[j] * r[j];      
	b = rrN / rr; rr = rrN;      
	for (int j=0; j<N; ++j) P[j] = r[j] + b * P[j];      
	++iter;
	if (iter > maxiter) {
	  printf("..ABORT: reached max iterations without converging\n");
	  break;
	}
      }
      printf("..dbg: conjugate gradient converged in %zu iterations\n", iter);
      for (int j=0; j<N; ++j) X[j] = x[j];

      // cleanup  
      if (r) { delete[] r; r = 0; }
      if (P) { delete[] P; P = 0; }
      if (x) { delete[] x; x = 0; }
      if (S) { delete[] S; S = 0; }    
    }

    template<typename T> void BiCGStab(const T * A, T * X, T * B, const int N) {
    
    }
  }; // end namespace Matrix
}; // end namespace Math

