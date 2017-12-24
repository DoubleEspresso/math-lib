#include <string>
#include <cstring>
#include <vector>

#include "export.h"
#include "threads.h"
#include "rand.h"
#include "complex.h"
#include "clock.h"

#define K(x) -M_PI + 2.0*M_PI*x

void test_mm();
void test_lu();
void test_qr();
void test_ge();
void test_vegas();
void test_dot();
void test_cgd(int N);

int main(int argc, char ** argv) {
  test_mm();
  test_lu();
  test_qr();
  test_ge();
  test_vegas();
  test_dot();
  test_cgd(100);
  return 0;
}

// lattice-qcd pertubation theory integrand for vegas algorithm
double z3(double * x, void * p) {
  // 0.107781313539874001343391550
  double den = 0;
  for (int j = 0; j < 4; ++j) den += sin(K(x[j]) / 2)*sin(K(x[j]) / 2);
  double A = sin(K(x[0]) / 2) * sin(K(x[0]) / 2) * sin(K(x[1]) / 2) * sin(K(x[1]) / 2);
  return A / den;
}

void test_cgd(int N) { 
  MT19937<double> rand(0.01, 1);
  int NN = N * N;
  // make a symmetric (positive definite), NxN matrix A
  // and a random matrix X, to compute B
  double * A = new double[NN]; memset(A, 0, NN * sizeof(double));
  double * X = new double[N]; memset(X, 0, N * sizeof(double));
  double * X0 = new double[N]; memset(X0, 0, N * sizeof(double));
  double * B = new double[N]; memset(B, 0, N * sizeof(double));
  double * Br = new double[N]; memset(Br, 0, N * sizeof(double));

  for (int r=0; r<N; ++r) {
    for (int c=0; c<=r; ++c) {
      int i1 = r * N + c;
      int i2 = c * N + r;
      A[i1] = rand.next();
      A[i2] = A[i1];
    }
  }
  for (int j=0; j<N; ++j) { A[j*N+j] = rand.next(); X[j] = rand.next(); }

  mm_cpup(A, X, B, N, N, 1); // compute B

  // find the solution Ax = b 
  // using a gradient descent method
  Clock c("cgd_naive");
  cgd_cpu(A, X0, B, N);
  c.finished();

  mm_cpup(A, X0, Br, N, N, 1); // compute B
  
  int nerr = 0; double tol = 1e-3;
  for (int j=0; j<N; ++j) {
    if(fabs(B[j]-Br[j]) > tol) ++nerr;
  }
  printf(".d.cgd(%dx%d) finished with %d-errors\n", N, N, nerr);
  
  // cleanup
  if (A) { delete[] A; A = 0; }
  if (X) { delete[] X; X = 0; }
  if (X0) { delete[] X0; X0 = 0; }
  if (B) { delete[] B; B = 0; }
  if (Br) { delete[] Br; Br = 0; }
}

void test_dot() {
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int> uniform_dist(2, 20);
  int npassed = 0;
  for (int i = 100; i < 110; ++i) {
    size_t dim = uniform_dist(e1);
    float * a = new float[dim];
    float * b = new float[dim];
    float ccpu = 0;
    float cgpu = 0;
    float ccpur = 0;
    
    for (int j = 0; j < dim; ++j) {
      a[j] = 0.0001*j; b[j] = 0.0001*j;
    }    
    dot_gpu(a, b, &cgpu, dim);
    dot_cpup(a, b, &ccpu, dim);
    dot_cpu(a, b, &ccpur, dim);
    
    std::string err_str = "PASSED";
    float e = ccpu - cgpu;
    if (e > 10 || (e < 0 && e < -10)) err_str = "FAILED";
    
    if (err_str == "FAILED") printf("..%zu %s %3.3f %3.3f\n", dim, err_str.c_str(), cgpu, ccpu);
    
    if (a) { delete[] a; a = 0; }
    if (b) { delete[] b; b = 0; }
  }
}


void test_vegas() {
#ifdef _WIN32
  LARGE_INTEGER frequency;        // ticks per second
  LARGE_INTEGER t1, t2;           // ticks
  double elapsedTime;
  QueryPerformanceFrequency(&frequency);  // get ticks per second
  QueryPerformanceCounter(&t1);	// start timer
#endif
  
  double xl[] = { 0, 0, 0, 0};
  double xu[] = { 1, 1, 1, 1 };
  //double xl[] = { -M_PI, -M_PI, -M_PI };
  //double xu[] = { M_PI, M_PI, M_PI };
  double res = 0; double err = 0;

  vegas_cpup(z3, 0, xl, xu, 4, 10000, &res, &err);
  printf("..(multi-thread vegas) res=%4.14f +/- %4.14f\n", res, err);

#ifdef _WIN32
  QueryPerformanceCounter(&t2);
  elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
  printf("..vegas_cpu(%3.1fms) (multi-thread)\n", (float)elapsedTime);

  QueryPerformanceFrequency(&frequency);  // get ticks per second
  QueryPerformanceCounter(&t1);	// start timer
#endif
  
  vegas_cpu(z3, 0, xl, xu, 4, 10000, &res, &err);
  printf("..(single-thread vegas) res=%4.14f +/- %4.14f\n", res, err);
  
#ifdef _WIN32
  QueryPerformanceCounter(&t2);
  elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
  printf("..vegas_cpu(%3.1fms) (single-thread)\n", (float)elapsedTime);
#endif
}

void test_ge() {
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int> uniform_dist(2, 800);
  int npassed = 0; int ntrials = 0;
  
  for (int i = 1; i < 10; ++i) {
    size_t N = i * uniform_dist(e1);
    if (N > 796) N = 796;

    printf("..size (%zu,%zu)\n", N, N);
    size_t dim = N*N;
    
    double * a = new double[dim];
    double * x = new double[N];
    double * b = new double[N];
    double * bans = new double[N];
    
    std::memset(a, double(0), dim * sizeof(double));
    std::memset(b, double(0), N * sizeof(double));
    std::memset(bans, double(0), N * sizeof(double));
    std::memset(x, double(0), N * sizeof(double));
    
    for (int r = 0; r < N; ++r) {
      int min = r - 1 < 0 ? 0 : r - 1;
      int max = r + 2 > N ? N : r + 2;
      for (int c = min; c < max; ++c) {
	a[r*N + c] = (double)((double)uniform_dist(e1) / (double)uniform_dist(e1));
      }
      x[r] = (double)((double)uniform_dist(e1) / (double)uniform_dist(e1));
    }    
    mm_cpup(a, x, bans, N, N, 1);
    
    
    //printf("\n============= b =============\n");
    for (int r = 0; r < N; ++r) {
      //printf("%3.3f ", bans[r]);
      b[r] = bans[r];
    }
    ge_cpu(a, b, N);
    
    // check
    bool pass = true;
    for (int r = 0; r < N; ++r) {
      double e = x[r] - b[r];
      if (e > 0.1 || (e < 0 && e < -0.1)) {
	pass = false; break;
      }
    }
    if (pass) ++npassed;
    else printf("..(%zu,%zu) %s\n", N, N, "FAILED");
    ++ntrials;
    
    if (a) { delete[] a; a = 0; }
    if (b) { delete[] b; b = 0; }
    if (bans) { delete[] bans; bans = 0; }
  }
  printf("..GE FINISHED %d of %d passed %3.3f%s\n", npassed, ntrials, 100.0 * ((double)npassed / (double)ntrials), "-acc");
}

void test_qr() {
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int> uniform_dist(2, 20);
  int npassed = 0; int ntrials = 0;

  for (int i = 40; i < 50; ++i) {
    size_t N = i * uniform_dist(e1);
    if (N > 4056) N = 4056;
    
    size_t dim = N*N;
    
    double * a = new double[dim];
    double * Q = new double[dim];
    double * R = new double[dim];
    double * ans = new double[dim];
    
    std::memset(a, double(0), dim * sizeof(double));
    std::memset(Q, double(0), dim * sizeof(double));
    std::memset(R, double(0), dim * sizeof(double));
    std::memset(ans, double(0), dim * sizeof(double));
    
    for (int r = 0, idx = 0; r < N; ++r) {
      for (int c = 0; c < N; ++c, ++idx) {
	a[idx] = (double)((double)uniform_dist(e1) / (double)uniform_dist(e1));
	R[idx] = a[idx];
      }
    }
      
    //printf("=== Q in ====\n");
    //for (int r = 0, idx = 0; r < N; ++r) {
    //	for (int c = 0; c < N; ++c, ++idx) {
    //		printf(" %3.3f ", Q[idx]);
    //	}
    //	printf("\n");
    //}

    //printf("=== R in ====\n");
    //for (int r = 0, idx = 0; r < N; ++r) {
    //	for (int c = 0; c < N; ++c, ++idx) {
    //		printf(" %3.3f ", R[idx]);
    //	}
    //	printf("\n");
    //}

    qr_cpu(a, Q, R, N);
    mm_cpup(Q, R, ans, N, N, N);
      
    //printf("=== Q out ====\n");
    //for (int r = 0, idx = 0; r < N; ++r) {
    //	for (int c = 0; c < N; ++c, ++idx) {
    //		printf(" %3.3f ", Q[idx]);
    //	}
    //	printf("\n");
    //}

    //printf("=== R out ====\n");
    //for (int r = 0, idx = 0; r < N; ++r) {
    //	for (int c = 0; c < N; ++c, ++idx) {
    //		printf(" %3.3f ", R[idx]);
    //	}
    //	printf("\n");
    //}

    size_t ecount = 0;
    std::string estr = "FAILED"; int track_idx = 0; double diff = 0;
    for (int idx = 0; idx < dim; ++idx) {
      double e = a[idx] - ans[idx];
      if (e > 0.1 || (e < 0 && e < -0.1)) {
	++ecount; track_idx = idx; diff = e;
      }
    }
    if (ecount <= 0) {
      estr = "PASSED"; ++npassed;
    }
    ++ntrials;
    if (ecount > 0) printf("..(%zu,%zu) %s with %zu-errors\n", N, N, estr.c_str(), ecount);
    else printf("..size (%zu,%zu) PASSED!\n", N, N);
  }
}

void test_lu() {
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int> uniform_dist(2, 800);
  int npassed = 0; int ntrials = 0;
  
  for (int i = 1; i < 3; ++i) {
    size_t N = i * uniform_dist(e1);
    if (N > 4056) N = 4056;
    
    size_t dim = N*N;
    
    double * a = new double[dim];
    double * al = new double[dim];
    double * au = new double[dim];
    double * bcpu = new double[dim];
    double * bgpu = new double[dim];

    std::memset(a, double(0), dim * sizeof(double));
    std::memset(al, double(0), dim * sizeof(double));
    std::memset(au, double(0), dim * sizeof(double));

    for (int r = 0, idx = 0; r < N; ++r) {
      for (int c = 0; c < N; ++c, ++idx) {
	if (c <= r) al[idx] = (c == r ? 1. : (double)((double)uniform_dist(e1) / (double)uniform_dist(e1)));//(c == r ? 1. : .1 + 0.05 * sin(r)*sin(r)*cos(c));
	if (c >= r) au[idx] = (double)((double)uniform_dist(e1) / (double)uniform_dist(e1));//.1 + 0.06 * cos(r) * sin(c) * sin(c);
	bcpu[idx] = 0.; bgpu[idx] = 0.;
      }
    }
    int * p = new int[N];
    for (int j = 0; j < N; ++j) p[j] = j;
    
    mm_cpup(al, au, a, N, N, N);      
    lu_gpu(a, bcpu, p, N);
    //for (int j = 0; j < N; ++j) printf(" %ds ", p[j]); printf("\n");

    double * P = new double[dim]; std::memset(P, 0., dim * sizeof(double));
    double * Pt = new double[dim]; std::memset(Pt, 0., dim * sizeof(double));
    for (int j = 0, idx = 0; j < N; ++j, idx += N) {
      P[idx + p[j]] = 1.;
    }
    for (int r = 0; r < N; ++r) {
      for (int c = 0; c < N; ++c) {
	Pt[r * N + c] = P[N*c + r];
      }
    }

    // split b into upper/lower triangular matrices
    double * u = new double[dim]; memset(u, 0., sizeof(double) * dim);
    double * l = new double[dim]; memset(l, 0., sizeof(double) * dim);
    double * res = new double[dim];
    double * res2 = new double[dim];
    for (int r = 0, idx = 0; r < N; ++r) {
      for (int c = 0; c < N; ++c, ++idx) { 
	if (c <= r) l[idx] = (c == r ? 1 : bcpu[idx]);
	if (c >= r) u[idx] = bcpu[idx];
	res[idx] = 0; res2[idx] = 0;
	//printf(" %3.3f ", u[idx]);
      }
      //printf("\n");
    }
    mm_cpup(l, u, res2, N, N, N);
    mm_cpup(Pt, res2, res, N, N, N);
    
    // check
    bool pass = true;
    for (int r = 0, idx = 0; r < N; ++r) {
      for (int c = 0; c < N; ++c, ++idx) {
	double e = res[idx] - a[idx];
	if (e > 0.1 || (e < 0 && e < -0.1)) {
	  pass = false; 
	  //printf("(%d,%d) %3.3f\n", r, c, e);
	  break;
	}
	//printf(" %3.3f ", res[idx]);
      }
      if (!pass) break;
      //printf("\n");
    }
    if (pass) ++npassed;
    else printf("..(%zu,%zu) %s\n", N, N, "FAILED");
    ++ntrials;

    if (P) { delete[] P; P = 0; }
    if (Pt) { delete[] Pt; Pt = 0; }
    if (res2) { delete[] res2; res2 = 0; }
    if (a) { delete[] a; a = 0; }
    if (p) { delete[] p; p = 0; }
    if (al) { delete[] al; al = 0; }
    if (au) { delete[] au; au = 0; }
    if (l) { delete[] l; l = 0; }
    if (u) { delete[] u; u = 0; }
    if (res) { delete[] res; res = 0; }
    if (bcpu) { delete[] bcpu; bcpu = 0; }
    if (bgpu) { delete[] bgpu; bgpu = 0; }
  }
  printf("..LU FINISHED %d of %d passed %3.3f%s\n", npassed, ntrials, 100.0 * ((double)npassed / (double)ntrials), "-acc");

}

void test_mm() {
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int> uniform_dist(16, 26);
  int npassed = 0; int ntrials = 0;
  for (int i = 2; i < 20; ++i) {
    // gpu error tial 1: ra > rb && ca < cb
    size_t Nr_a = i * uniform_dist(e1);
    size_t Nc_a = Nr_a; // i * uniform_dist(e1);
    size_t Nr_b = Nc_a;
    size_t Nc_b = Nr_a; // i * uniform_dist(e1);
    char * a = new char[Nr_a * Nc_a];
    char * b = new char[Nr_b * Nc_b];
    char * ccpu = new char[Nr_a * Nc_b];
    char * cgpu = new char[Nr_a * Nc_b];

    for (unsigned int j = 0; j < Nr_a * Nc_a; ++j) a[j] = (char)j;// 0.0001 * j;
    for (unsigned int j = 0; j < Nr_b * Nc_b; ++j) b[j] = (char)j;// 0.0002 * j;
    for (unsigned int j = 0; j < Nr_a * Nc_b; ++j) {
      ccpu[j] = 0; cgpu[j] = 0;
    }

    mm_cpu(a, b, cgpu, Nr_a, Nc_a, Nc_b);
    //mm_cpup(a, b, ccpu, Nr_a, Nc_a, Nc_b);
    mm_gpu(a, b, ccpu, Nr_a, Nc_a, Nc_b);
    
    //Math::Matrix::axb(a, b, ccpu, Nr_a, Nc_a, Nc_b);
    //mm_cpu(a, b, ccpu, Nr_a, Nc_a, Nc_b);

    size_t ecount = 0;
    std::string estr = "FAILED";
    for (unsigned int idx = 0; idx < Nr_a * Nc_b; ++idx) {
      double e = ccpu[idx] - cgpu[idx];
      if (e > 0.1 || (e < 0 && e < -0.1)) {
	++ecount;
      }
    }
    if (ecount <= 0) {
      estr = "PASSED"; ++npassed;
    }
    ++ntrials;
    if (ecount > 0) printf("..(%zu,%zu) %s with %zu-errors\n", Nr_a, Nc_b, estr.c_str(), ecount);

    if (a) { delete[] a; a = 0; }
    if (b) { delete[] b; b = 0; }
    if (ccpu) { delete[] ccpu; ccpu = 0; }
    if (cgpu) { delete[] cgpu; cgpu = 0; }
  }

  printf("..FINISHED %d of %d passed %3.3f%s\n", npassed, ntrials, 100.0 * ((double) npassed / (double) ntrials),"-acc");
}
