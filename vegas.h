#pragma once

#ifndef MATHLIB_VEGAS_H
#define MATHLIB_VEGAS_H

#include <stdlib.h>
#include "rand.h"

#ifdef MATHLIB_EXPORTS  
#define MATHLIB_API __declspec(dllexport)   
#else  
#define MATHLIB_API __declspec(dllimport)   
#endif  

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

class vegas {
	size_t MAXBINS;
	state * s;
	size_t dim;
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
	void tracegrid();
	void dbg_grid(int iter);

public:
	vegas(size_t _d) : s(0), r(0), dim(_d), MAXBINS(50) { init(); }
	~vegas() { free(); }

	int integrate(vegas_integrand f, void * params, double xl[], double xu[], double * res, double * abserr);
	double chi_sq() { return s->chisq; }
	void set_maxbins(size_t b);
	void set_evals(size_t e) { s->evals = e; }
};

// external calls
namespace Math
{
	class Integrate
	{
	public:
		static MATHLIB_API void Vegas(vegas_integrand f, void * params, double xl[], double xu[], size_t dim, size_t icalls, double * res, double * abserr, bool gpu = false);
	};
}
#endif
