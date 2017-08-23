#include "vegas2.h"

#include <cmath>
#include <stdio.h>
#include <cstring>
#include <vector>

#define MAX(a,b) (a > b ? a : b)
#define MIN(a,b) (a > b ? b : a)

void vegas2::init(size_t dim) {
	if (s == 0) s = new vegas2_state();
	if (r == 0) r = new MT19937<double>(0, 1);
	s->dx = new double[dim]; std::memset(s->dx, 0, dim*sizeof(double));
	s->hist = new double[BINS_MAX*dim]; std::memset(s->hist, 0, BINS_MAX*dim*sizeof(double));
	s->xt = new double[(BINS_MAX + 1)*dim]; std::memset(s->xt, 0, (BINS_MAX + 1)*dim*sizeof(double));
	s->w = new double[BINS_MAX]; std::memset(s->w, 0, BINS_MAX*sizeof(double));
	s->bidx = new int[dim]; std::memset(s->bidx, 0, dim*sizeof(int));
	s->box = new int[dim]; std::memset(s->box, 0, dim*sizeof(int));
	s->x = new double[dim]; std::memset(s->x, 0, dim*sizeof(double));
	
	s->dim = dim;
	s->stage = 0;
	s->alpha = 1.5;
	s->iters = 5;
	s->mode = IMPORTANCE;
	s->chisq = 0;
	s->bins = BINS_MAX;
	s->jac = 0;
	s->wint_sum = 0; 
	s->w_sum = 0;
	s->chi_sum = 0;
	s->samples = 0;
	s->evals_per_iter = 0;
	s->result = 0;
	s->sigma = 0;
}

void vegas2::free() {
	if (s->dx) { delete[] s->dx; s->dx = 0; }
	if (s->hist) { delete[] s->hist; s->hist = 0; }
	if (s->xt) { delete[] s->xt; s->xt = 0; }
	if (s->w) { delete[] s->w; s->w = 0; }
	if (s->bidx) { delete[] s->bidx; s->bidx = 0; }
	if (s->box) { delete[] s->box; s->box = 0; }
	if (s->x) { delete[] s->x; s->x = 0; }
	if (s) { delete s; s = 0; }
	if (r) { delete r; r = 0; }
}

int vegas2::integrate(vegas2_integrand f, void * params, double xl[], double xu[], size_t icalls, double* result, double* abserr) {

	if (s->stage == 0) {
		s->vol = 1.0; s->bins = 1;
		for (unsigned int j = 0; j < s->dim; ++j) {
			s->xt[j] = 0.0; s->xt[s->dim + j] = 1.0;
			s->dx[j] = xu[j] - xl[j]; s->vol *= s->dx[j];
		}
	}
	if (s->stage <= 1) {
		s->wint_sum = 0;
		s->w_sum = 0;
		s->chi_sum = 0;
		s->samples = 0;
		s->chisq = 0;
	}
	unsigned int bins = BINS_MAX;
	unsigned int evls = floor(pow(icalls / 2.0, 1.0 / s->dim));
	s->mode = IMPORTANCE;
	if (BINS_MAX <= 2 * evls) {
		int evls_per_bin = MAX(evls / BINS_MAX, 1);
		bins = MIN(evls / evls_per_bin, BINS_MAX);
		evls = evls_per_bin * bins;
		s->mode = STRATIFIED;
	}
	double tot_evls = pow((double)evls, (int)s->dim);
	s->evals_per_iter = MAX(icalls / tot_evls, 2);
	icalls = s->evals_per_iter  * tot_evls;
	s->jac = s->vol * pow((double)bins, (double)s->dim) / icalls;
	//printf("..scaling=%4.14f\n", pow((double)bins, (double)s->dim) / icalls);
	s->evals = evls;
	if (bins != s->bins) resize_grid(bins);

	double tot_int = 0.0; double tot_sig = 0.0;
	for (unsigned int it = 0; it < s->iters; ++it) {
		double intg = 0.0, tss = 0.0, wgt = 0.0;

		// clear
		for (unsigned int i = 0; i < s->bins; ++i) {
			for (unsigned int j = 0; j < s->dim; ++j) {
				s->hist[s->dim*i + j] = 0;
			}
		}
		for (unsigned int i = 0; i < s->dim; ++i) s->box[i] = 0;

		do {
			double m = 0, q = 0, fsq_sum = 0;
			for (unsigned int k = 0; k < s->evals_per_iter; ++k) {
				double bin_vol = 0; rand_x(xl, bin_vol); // initializes x-array
				double fval = s->jac * bin_vol * f(s->x, params);
				double d = fval - m;
				m += d / (k + 1.0);
				q += d*d*(k / (k + 1.0));
				if (s->mode != STRATIFIED) accumulate(fval*fval);
			}
			intg += m * s->evals_per_iter;
			fsq_sum = q * s->evals_per_iter;
			tss += fsq_sum;
			if (s->mode == STRATIFIED) accumulate(fsq_sum);
		} while (adjust());

		double var = tss / (s->evals_per_iter - 1.0);
		double intg_sq = intg * intg;

		if (var > 0) wgt = 1.0 / var;
		else if (s->w_sum > 0) wgt = s->w_sum / s->samples;
		else wgt = 0;

		s->sigma = sqrt(var);
		s->result = intg;

		if (wgt > 0) {
			double m = (s->w_sum > 0 ? s->wint_sum / s->w_sum : 0);
			double q = intg - m;
			s->samples++;
			s->w_sum += wgt;
			s->wint_sum += intg * wgt;
			s->chi_sum += intg_sq * wgt;
			tot_int = s->wint_sum / s->w_sum;
			tot_sig = sqrt(1 / s->w_sum);

			if (s->samples == 1) s->chisq = 0;
			else {
				s->chisq *= (s->samples - 2.0);
				s->chisq += (wgt / (1 + (wgt / s->w_sum))) * q * q;
				s->chisq /= (s->samples - 1.0);
			}
		}
		else {
			tot_int += (intg - tot_int) / (it + 1.0);
			tot_sig = 0.0;
		}
		refine_grid();
	}
	s->stage = 1; // note: stage never increases beyond 1
	*result = tot_int;
	*abserr = tot_sig;
	return 1;
}

bool vegas2::adjust() {
	for (int j = s->dim - 1; j >= 0; --j) {
		s->box[j] = (s->box[j] + 1) % s->evals;
		if (s->box[j] != 0) return true;
	}
	return false;
}

void vegas2::resize_grid(unsigned int bins) {
	double w = (double)s->bins / (double)bins;

	for (unsigned int j = 0; j < s->dim; j++) {
		double xold = 0; double xnew = 0; double dw = 0;
		std::vector<double> xtmp;
		for (unsigned int k = 1; k <= s->bins; ++k) {
			dw += 1.0;
			xold = xnew; xnew = s->xt[s->dim*k + j];
			while (dw > w) {
				dw -= w;
				xtmp.push_back(xnew - (xnew - xold)*dw);
			}
		}
		for (unsigned int k = 1; k < bins; k++) s->xt[s->dim*k + j] = xtmp[k - 1];
		s->xt[s->dim*bins + j] = 1;
	}
	s->bins = bins;
}

void vegas2::rand_x(double xl[], double& binvol) {
	binvol = 1.0; size_t d = s->dim;
	for (unsigned int j = 0; j < d; ++j) {
		double random = 0;
		while (random == 0 || random == 1) random = r->next();
		double z = ((s->box[j] + random) / s->evals) * s->bins; // map to bin space
		int k = z; double bwidth = 0; double y = 0;
		s->bidx[j] = k;
		if (k == 0) {
			bwidth = s->xt[d + j];
			y = z * bwidth;
		}
		else {
			bwidth = s->xt[(k + 1)*d + j] - s->xt[k*d + j];
			y = s->xt[k*d + j] + (z - k)*bwidth; // map from bin space to coord space
		}
		s->x[j] = xl[j] + y * s->dx[j]; // map from coord space to domain
		binvol *= bwidth;
	}
}

void vegas2::accumulate(double y) {
	for (unsigned int j = 0; j < s->dim; j++) s->hist[s->bidx[j] * s->dim + j] += y;
}

void vegas2::refine_grid() {
	for (unsigned int j = 0; j < s->dim; j++) {
		double oh = s->hist[j];
		double nh = s->hist[s->dim + j];
		s->hist[j] = (oh + nh) / 2;
		double grid_tot_j = s->hist[j];

		for (unsigned int i = 1; i < s->bins - 1; ++i) {
			double tmp = oh + nh;
			oh = nh;
			nh = s->hist[(i + 1)*s->dim + j];
			s->hist[i*s->dim + j] = (tmp + nh) / 3;
			grid_tot_j += s->hist[i*s->dim + j];
		}

		s->hist[(s->bins - 1)*s->dim + j] = (nh + oh) / 2;
		grid_tot_j += s->hist[(s->bins - 1)*s->dim + j];
		double tot_weight = 0;
		for (unsigned int i = 0; i < s->bins; i++) {
			s->w[i] = 0;
			if (s->hist[s->dim*i + j] > 0) {
				oh = grid_tot_j / s->hist[s->dim*i + j];
				// damped change
				s->w[i] = pow(((oh - 1) / oh / log(oh)), s->alpha);
			}
			tot_weight += s->w[i];
		}

		double pts_per_bin = tot_weight / s->bins;
		double xold = 0; double xnew = 0; double dw = 0;
		std::vector<double> xtmp;
		for (unsigned int k = 0; k < s->bins; k++) {
			dw += s->w[k];
			xold = xnew; xnew = s->xt[(k + 1)*s->dim + j];

			while (dw > pts_per_bin) {
				dw -= pts_per_bin;
				xtmp.push_back(xnew - (xnew - xold)*dw / s->w[k]);
			}
		}
		for (unsigned int k = 1; k < s->bins; ++k) s->xt[k*s->dim + j] = xtmp[k - 1];
		s->xt[s->bins*s->dim + j] = 1;
	}
}

void vegas2::tracegrid() {
	for (unsigned int j = 0; j < s->dim; ++j) {
		printf("\n axis %lu \n", j);
		printf("      x   \n");
		for (unsigned int i = 0; i <= s->bins; i++) {
			printf("%11.2e", s->xt[s->dim*i + j]);
			if (i % 5 == 4) printf("\n");
		}
		printf("\n");
	}
	printf("\n");
}

// dll call
namespace Math
{
	extern "C" void Integrate::vegas(vegas2_integrand f, void * params, double xl[], double xu[], size_t dim, size_t icalls, double * res, double * abserr, bool gpu)
	{
		vegas2 v(dim);
		v.integrate((vegas2_integrand)f, params, xl, xu, 10000, res, abserr);
		do {
			v.integrate((vegas2_integrand)f, params, xl, xu, icalls, res, abserr);
		} while (fabsf(v.chi_sq() - 1) > 0.5);
	}
}
