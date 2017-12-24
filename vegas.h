#pragma once
#ifndef MATHLIB_VEGAS_H
#define MATHLIB_VEGAS_H

#include <cstring>
#include <stdlib.h>

#include "rand.h"
#include "threads.h"

#define BINS_MAX 50
typedef double(*vegas_integrand)(double*, void*);
enum { IMPORTANCE = 0, STRATIFIED = 1 };

struct state {
  size_t bins, iters, evals;
  double * xi; // interval markings sz=bins*dim, range:[0,1]
  int * xid; // utility array tracking xi index 
  double * w; // weighting factors 
  double * hist; // histogrammed function values
  double alpha; // stiffness parameter for bin refinement
  int stage, type;
  double result, wsum, chisq, vol;
};

struct X {
public:
X(int * d, size_t s) : x(0), sz(s) { 
  x = new int[sz]; 
  for (unsigned int j = 0; j < sz; ++j) x[j] = d[j];
}  
  ~X() { if (x) { delete[] x; x = 0; } }
  
X(const X& xc) : x(0) {
  if (x) { delete[] x; x = 0; }
  x = new int[xc.sz];
  sz = xc.sz;
  memset(x, 0, sz*sizeof(int));
  for (unsigned int j = 0; j < sz; ++j) x[j] = xc.x[j];
} 
  double val(int i) { return x[i]; }

private:
  int * x;
  size_t sz;
};

struct vegas_tdata {
  unsigned long tid, dim, nb_evals, bins;
  double fval, jac; // average function eval over bins
  double vsum; // pooled variance over bins
  double * xl;
  double * dx;
  double * xi;
  double * t_hist; // thread histogram
  MT19937<double> * r;
  std::vector<X> x;
  vegas_integrand f;
  void * params;
};

class vegas {
  size_t MAXBINS;
  state * s;
  size_t dim, nb_threads;
  THREAD_HANDLE * threads;
  std::vector<vegas_tdata> tdata;
  MT19937<double> *r;
  void init();
  void free();
  void rand_x(double xl[], double dx[], double * x, double& binvol);
  void accumulate(double y);
  void resize_grid();
  void refine_grid();
  double smooth();
  void clear();
  bool adjust();
  void init_threads(double jac, vegas_integrand f, void * params, double xl[], double dx[]);
  void tracegrid();
  void dbg_grid(int iter);

 public:
 vegas(size_t _d) : s(0), r(0), dim(_d), MAXBINS(50), threads(0), nb_threads(4) { init(); }
  ~vegas() { free(); }
  void set_threads(size_t nb);
  int pintegrate(vegas_integrand f, void * params, double xl[], double xu[], double * res, double * abserr);
  int integrate(vegas_integrand f, void * params, double xl[], double xu[], double * res, double * abserr);
  double chi_sq() { return s->chisq; }
  void set_maxbins(size_t b);
  void set_evals(size_t e) { s->evals = e; }
};


namespace Math {
  namespace Integrate {
    void Vegas(vegas_integrand f, void * params, double xl[], double xu[], size_t dim, size_t icalls, double * res, double * abserr);
    void Vegasp(vegas_integrand f, void * params, double xl[], double xu[], size_t dim, size_t icalls, double * res, double * abserr);
  };
};

#endif
