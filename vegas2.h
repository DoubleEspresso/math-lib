#pragma once

#ifndef MATHLIB_VEGAS2_H
#define MATHLIB_VEGAS2_H

#include <stdlib.h>
#include "rand.h"

#ifdef MATHLIB_EXPORTS  
#define MATHLIB_API __declspec(dllexport)   
#else  
#define MATHLIB_API __declspec(dllimport)   
#endif  

#define BINS_MAX 50
typedef double (*vegas2_integrand)(double*, void*);

enum { IMPORTANCE = 0, STRATIFIED = 1 };

typedef struct {
	size_t dim;
	unsigned int bins, evals;
	double * xt;  // transformed coordinates range:[0,1]
	double * w; // weighting factors 
	double * dx; // integration intervals in coord space
	double vol;
	double * x; // untransformed coordinates range:[a,b]
	int * bidx; // pointer to current bin 
	int * box; // pointer to current interval along each dim
	double * hist; // histogrammed function values

	double jac, wint_sum, w_sum, chi_sum, chisq;
	unsigned int samples, evals_per_iter;
	double alpha;
	int mode, stage;
	unsigned int iters;
	double result, sigma;
} vegas2_state;

class vegas2 {
	vegas2_state *s;
	MT19937<double> *r;
	void init(size_t dim);
	void free();
	void rand_x(double xl[], double& binvol);
	void accumulate(double y);
	void resize_grid(unsigned int bins);
	void refine_grid();
	bool adjust();
	void tracegrid();

public:
	vegas2() : s(0), r(0) { }
	vegas2(size_t _d) : s(0), r(0) { init(_d); }
	~vegas2() { free(); }

	int integrate(vegas2_integrand f, void * params, double xl[], double xu[], size_t icalls, double * res, double * abserr);
	double chi_sq() { return s->chisq; }
};

// external calls
namespace Math
{
	class Integrate
	{
	public:
		static MATHLIB_API void vegas(vegas2_integrand f, void * params, double xl[], double xu[], size_t dim, size_t icalls, double * res, double * abserr, bool gpu = false);
	};
}
#endif
