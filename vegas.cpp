#include <cmath>
#include <stdio.h>
#include <cstring>
#include <vector>
#include <fstream>

#include "vegas.h"

#define MAX(a,b) (a > b ? a : b)
#define MIN(a,b) (a > b ? b : a)

void vegas::init() {
  if (s == 0) s = new state();
  if (r == 0) r = new MT19937<double>(0, 1);
  s->xi = new double[(MAXBINS + 1)*dim]; memset(s->xi, 0, (MAXBINS + 1)*dim*sizeof(double));
  s->xid = new int[dim]; memset(s->xid, 0, dim*sizeof(int));
  s->hist = new double[MAXBINS*dim]; memset(s->hist, 0, MAXBINS*dim*sizeof(double));
  s->w = new double[MAXBINS]; memset(s->w, 0, MAXBINS*sizeof(double));
  s->stage = 0;
  s->alpha = 1.5;
  s->iters = 5;
  s->evals = 2;
  s->type = IMPORTANCE;
  s->chisq = 0;
  s->bins = MAXBINS;
  s->result = 0;
  s->wsum = 0;
}

void vegas::set_maxbins(size_t b) {
  free(); MAXBINS = b; init();
}

void vegas::free() {
  if (s->hist) { delete[] s->hist; s->hist = 0; }
  if (s->w) { delete[] s->w; s->w = 0; }
  if (s->xi) { delete[] s->xi; s->xi = 0; }
  if (s->xid) { delete[] s->xid; s->xid = 0; }
  if (s) { delete s; s = 0; }
  if (r) { delete r; r = 0; }
  if (threads) { delete[] threads; threads = 0; }
}

void vegas::set_threads(size_t nb) {
  if (threads) { delete[] threads; threads = 0; }
  nb_threads = nb;
  if (threads == 0) threads = new THREAD_HANDLE[nb];
}

void work_task(void * p) {
  vegas_tdata * pp = (vegas_tdata*)p;
  pp->fval = 0; pp->vsum = 0; // init
  size_t bins = pp->bins + 1;
  double * loc_xi = pp->xi;
  double * x = new double[pp->dim]; memset(x, 0, pp->dim*sizeof(double));
	
  for (int j = 0; j < pp->x.size(); ++j) { // loop the index-set
    double m = 0; double q = 0; // mean and variance estimates for this pt
    for (unsigned int e = 0; e < pp->nb_evals; ++e) { // nb evals for this pt			
      double binvol = 1;
      for (unsigned int i = 0; i < pp->dim; ++i) { // compute an x-value
	double random = 0;
	while (random == 0 || random == 1) random = pp->r->next();
	int sidx = (int)(i * (pp->bins + 1) + pp->x[j].val(i));
	double lo = loc_xi[sidx];
	double hi = loc_xi[sidx + 1];
	double bwidth = hi - lo;
	//printf("!!DBG dim(%d), idx(%d), rand(%3.5f), lo(%3.5f), hi(%3.5f), bwidth(%3.5f)\n",
	//	i, sidx, random, lo, hi, bwidth);
	x[i] = pp->xl[i] + (lo + bwidth * random)*pp->dx[i];
	binvol *= bwidth;
      }
      double yval = pp->jac * binvol * pp->f(x, pp->params);
      //printf("!!DBG jac(%3.5f), binvol(%3.5f), x(%3.5f,%3.5f,%3.5f,%3.5f), func(%3.5f), y(%3.5f)\n",
      //	pp->jac, binvol, x[0], x[1], x[2], x[3], 
      //	pp->f(x, pp->params), yval);
      double d = yval - m;
      m += d / (e + 1.0); // favg
      q += d * (yval - m); // variance
      //if (s->type != STRATIFIED)
      {
	for (unsigned int h = 0; h < pp->dim; ++h)
	  {
	    int idx = (int)(h * (bins - 1) + pp->x[j].val(h));
	    pp->t_hist[idx] += yval*yval;
	  }
      }
    }
    pp->fval += m;
    pp->vsum += q;
  }
  if (x) { delete[] x; x = 0; }
}

void vegas::init_threads(double jac, vegas_integrand f, void * params, double xl[], double dx[]) {
  if (threads == 0) set_threads(nb_threads);
  size_t totbins = (size_t)(pow(s->bins, dim));
  int tstride = (int)(totbins / nb_threads);
  int trem = (int)(totbins - tstride * nb_threads); // remainder
  tdata.clear();

  int * xt = new int[dim]; memset(xt, 0, dim*sizeof(int));
  for (unsigned int t = 0, sidx = 0; t < nb_threads; ++t, sidx += tstride) {
    vegas_tdata d;
    d.tid = t;
    d.jac = jac;
    d.nb_evals = (unsigned long)s->evals;
    d.f = (vegas_integrand)f;
    d.params = (void*)params;
    d.xl = xl;
    d.dx = dx;
    d.bins = (unsigned long)s->bins;
    d.dim = (unsigned long)dim;
    d.t_hist = new double[MAXBINS*dim]; memset(d.t_hist, 0, MAXBINS*dim*sizeof(double));
    d.r = new MT19937<double>(0, 1);
    int mx = (tstride + (t == nb_threads - 1 ? trem : 0));
    for (int i = 0; i < mx; ++i) {
      for (size_t d = dim - 1; d >= 0; --d) {
	xt[d] = (xt[d] + 1) % s->bins;
	if (xt[d] != 0) break;
      }
      X pt(xt, dim); d.x.push_back(pt);
    }
    d.xi = s->xi;
    d.fval = d.vsum = 0;
    tdata.push_back(d);
  }
  if (xt) { delete[] xt; xt = 0; }
}

int vegas::pintegrate(vegas_integrand f, void * params, double xl[], double xu[], double * result, double * abserr) {
  double vol = 1.0, jac = 1.0;
  double * dx = new double[dim]; memset(dx, 0, dim*sizeof(double));
  double mean = 0, var = 0, isum = 0;

  for (unsigned int j = 0; j < dim; ++j) {
    dx[j] = xu[j] - xl[j];
    vol *= dx[j];
  }
  jac = vol;

  if (s->stage == 0) {
    for (unsigned int j = 0; j < dim; ++j) s->xi[j*(s->bins + 1) + 1] = 1.0;
    resize_grid();
    init_threads(jac, f, params, xl, dx);
  }
  else {
    for (int j = 0; j < nb_threads; ++j) {
      tdata[j].nb_evals = (unsigned long)s->evals; // reset
      tdata[j].jac = jac; 
      tdata[j].dx = dx;
      tdata[j].xl = xl;
    }
  }
  s->chisq = 0; s->wsum = 0; *result = 0;
  //printf("\t!!DBG start integrating..\n");
  for (unsigned int it = 0; it < s->iters; ++it) {
    double var = 0; double intg = 0; clear();

    for (int j = 0; j < nb_threads; ++j) {
      tdata[j].fval = 0;
      tdata[j].vsum = 0;
      tdata[j].xi = s->xi;
      memset(tdata[j].t_hist, 0, dim*s->bins * sizeof(double));
    }

    /*launch threads*/
    for (int j = 0; j < nb_threads; ++j) threads[j] = start_thread((thread_fnc)work_task, (void*)&tdata[j], j);
    wait_threads_finish(threads, (int)nb_threads);

    for (int j = 0; j < nb_threads; ++j)
      {
	intg += tdata[j].fval;
	var += tdata[j].vsum;
	for (unsigned int d = 0; d < dim; ++d)
	  {
	    for (int k = 0; k < tdata[j].x.size(); ++k) {
	      int idx = (int)(d * s->bins + tdata[j].x[k].val(d));
	      s->hist[idx] += tdata[j].t_hist[idx];
	    }
	  }
      }

    double d = (intg - mean);
    mean += d / (it + 1.0); // running average value of integral

    var /= (s->evals * (s->evals - 1)); // pooled variance from all threads

    if (var > 0) {
      double w = (intg * intg / var); // weight factor (this iteration)
      *result += intg * w; // numerator of weighted average
      s->wsum += w; // denominator of weighted average
      double wa = *result / s->wsum; // current weighted averag
      isum += intg * intg * w;

      double cs = (isum / (wa * wa) - s->wsum);
      s->chisq = (it > 1 ? cs / (it - 1.0) : cs);
    }
    else *result = mean;
    refine_grid();
    //printf("!!DBG result(%d)=%3.14f, chisq=%4.5f\n", it, *result / s->wsum, s->chisq);
  }
  s->stage = 1;
  *result /= s->wsum;
  *abserr = *result / (s->wsum > 0 ? sqrt(s->wsum) : 1);

  if (dx) { delete[] dx; dx = 0; }
  return 1;
}

int vegas::integrate(vegas_integrand f, void * params, double xl[], double xu[], double* result, double* abserr) {
  double vol = 1.0, jac = 1.0;
  double * x = new double[dim]; memset(x, 0, dim*sizeof(double));
  double * dx = new double[dim]; memset(dx, 0, dim*sizeof(double));
  double mean = 0, var = 0, isum = 0;

  if (s->stage == 0) {
    for (unsigned int j = 0; j < dim; ++j) s->xi[j*(s->bins + 1) + 1] = 1.0;
    resize_grid();
  }
  s->chisq = 0; s->wsum = 0; *result = 0;

  for (unsigned int j = 0; j < dim; ++j) {
    dx[j] = xu[j] - xl[j];
    vol *= dx[j];
  }
  jac = vol;

  for (unsigned int it = 0; it < s->iters; ++it) {
    double intg = 0, vsum = 0; clear();
    do {
      double m = 0, q = 0; // mean and variance tracking per bin over the whole domain of integration
      for (int i = 0; i < s->evals; ++i) {
	double bin_vol = 0;
	rand_x(xl, dx, x, bin_vol); // initializes x-array
	double fval = jac * bin_vol * f(x, params);

	// update the mean and variance of the integral using this
	// bin result (online recursion)
	double d = fval - m;
	m += d / (i + 1.0); // favg
	q += d * (fval - m); // variance
	if (s->type != STRATIFIED) accumulate(fval * fval);
      }
      intg += m; // nb. intg is now the sum of the average value of "f" in each bin
      vsum += q; // vsum is the sum of the variance of each bin over the whole domain
    } while (adjust());

    double d = (intg - mean);
    mean += d / (it + 1.0); // running average value of integral
    var = vsum / (s->evals*(s->evals - 1)); // the pooled bin-variance for this iteration

    if (var > 0) {
      double w = (intg * intg / var); // weight factor (this iteration)
      *result += intg * w; // numerator of weighted average
      s->wsum += w; // denominator of weighted average
      double wa = *result / s->wsum; // current weighted averag

      // chi-squared computation
      // note : this difference is unstable and prone to cancellations/rounding errors
      isum += intg * intg * w;
      double cs = (isum / (wa * wa) - s->wsum);
      s->chisq = (it > 1 ? cs / (it - 1.0) : cs);
    }
    else *result = mean;
    refine_grid();
  }
  s->stage = 1;
  *result /= s->wsum;
  *abserr = *result / (s->wsum > 0 ? sqrt(s->wsum) : 1);

  if (x) { delete[] x; x = 0; }
  if (dx) { delete[] dx; dx = 0; }
  return 1;
}

void vegas::accumulate(double y) {
  for (unsigned int j = 0; j < dim; j++) s->hist[j * s->bins + s->xid[j]] += y;
}

// clear the histogram and reset the starting point in the domain
void vegas::clear() {
  for (size_t i = 0, mx = s->bins * dim; i < mx; ++i) {
    s->hist[i] = 0;
    if (i < dim) s->xid[i] = 0;
  }
}

void vegas::resize_grid() {
  double w = (double)1.0 / (double)s->bins;
  for (unsigned int d = 0, idx = 0; d < dim; ++d) {
    for (unsigned int b = 0; b <= s->bins; ++b, ++idx) {
      s->xi[idx] = b*w;
    }
  }
}

bool vegas::adjust() {
  for (size_t j = dim - 1; j >= 0; --j) {
    s->xid[j] = (s->xid[j] + 1) % s->bins;
    if (s->xid[j] != 0) return true;
  }
  return false;
}

void vegas::rand_x(double xl[], double dx[], double * x, double& binvol) {
  binvol = 1.0;
  for (unsigned int d = 0; d < dim; ++d) {
    double random = 0;
    while (random == 0 || random == 1) random = r->next();
    int idx = s->xid[d];
    double lo = s->xi[d * (s->bins + 1) + idx];
    double hi = s->xi[d * (s->bins + 1) + (idx + 1)];
    double bwidth = hi - lo;
    x[d] = xl[d] + (lo + bwidth * random)*dx[d];
    binvol *= bwidth;
  }
}

double vegas::smooth() {

  // smooth the average before returning the sum  
  /*
    s->hist[0] = (s->hist[0] + s->hist[1]) / 2.0;
    double sum = s->hist[0];
    unsigned int mx = dim*s->bins-1;
    for (unsigned int i = 1; i < mx; ++i) {
    double av = (s->hist[i-1] + s->hist[i] + s->hist[i+1]) / 3.0;
    s->hist[i] = av;
    sum += av;
    }
    s->hist[mx] = (s->hist[mx-1] + s->hist[mx])/ 2.0;
    sum += s->hist[mx];
  */
  // direct sum, no smoothing  
  double sum = 0;
  for (size_t i = 0, mx = dim * s->bins; i < mx; ++i) {
    sum += s->hist[i];
  }
  return sum;
}

void vegas::refine_grid() {
  for (unsigned int d = 0, idx = 0; d < dim; ++d) {
    double hsum = smooth(); double wsum = 0;
    for (unsigned int i = 0; i < s->bins; ++i, ++idx) {
      s->w[i] = 0;
      if (s->hist[idx] > 0) {
	double v = s->hist[idx] / hsum; // normalized hist-value
	s->w[i] = pow((v - 1) / log(v), s->alpha); // update (see Lepage's original paper)
      }
      wsum += s->w[i];
    }

    double w_per_bin = wsum / s->bins;
    double xold = 0; double xnew = 0; double dw = 0; std::vector<double> x;
    for (unsigned int k = 0; k < s->bins; ++k) {
      dw += s->w[k];
      xold = xnew; xnew = s->xi[d*(s->bins + 1) + k + 1];
      while (dw > w_per_bin) {
	dw -= w_per_bin;
	x.push_back(xnew - (xnew - xold)*dw / s->w[k]);
      }
    }
    for (unsigned int b = 1; b < s->bins; ++b) s->xi[d*(s->bins + 1) + b] = x[b - 1];
  }
}

void vegas::tracegrid() {
  for (unsigned int j = 0, idx = 0; j < dim; ++j) {
    printf("\n axis %u \n", j);
    printf("      x   \n");
    for (unsigned int i = 0; i <= s->bins; ++i, ++idx) {
      printf("%11.2e", s->xi[idx]);
    }
    printf("\n");
  }
  printf("\n");
}

void vegas::dbg_grid(int iter) {
  char fname[256]; std::sprintf(fname, "grid_%d.txt", iter);

  std::ofstream outfile(fname, std::ofstream::binary);

  for (unsigned int b = 0; b <= s->bins; ++b) {
    for (unsigned int j = 0; j < dim; ++j) {
      outfile << s->xi[j*(s->bins + 1) + b] << "\t";
    }
    outfile << "\n";
  }
}

namespace Math {
  namespace Integrate {
    void Vegas(vegas_integrand f,
	       void * params,
	       double xl[],
	       double xu[],
	       size_t dim,
	       size_t icalls,
	       double * res,
	       double * abserr) {
      vegas v(dim);
      v.set_maxbins(10); v.set_evals(2);
      v.integrate(f, params, xl, xu, res, abserr);
      v.set_evals(16);
    
      do { v.integrate(f, params, xl, xu, res, abserr); }
      while (fabsf((float)(v.chi_sq() - 1)) > 0.5f);    
    }
  
    void Vegasp(vegas_integrand f,
		void * params,
		double xl[],
		double xu[],
		size_t dim,
		size_t icalls,
		double * res,
		double * abserr) {
      vegas v(dim);
      v.set_maxbins(10); v.set_evals(2);
      v.pintegrate(f, params, xl, xu, res, abserr);
      v.set_evals(16);
    
      do { v.pintegrate(f, params, xl, xu, res, abserr); }
      while (fabsf((float)(v.chi_sq() - 1)) > 0.5f);    
    }
  }
}
