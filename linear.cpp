#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include "linear.h"
#include "tron.h"
//TODO library for oursolver alone
#include <random>
#include <vector>

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void print_null(const char *s) {}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

class sparse_operator
{
public:
	static double nrm2_sq(const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += x->value*x->value;
			x++;
		}
		return (ret);
	}

	static double dot(const double *s, const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += s[x->index-1]*x->value;
			x++;
		}
		return (ret);
	}

	static void axpy(const double a, const feature_node *x, double *y)
	{
		while(x->index != -1)
		{
			y[x->index-1] += a*x->value;
			x++;
		}
	}

	static double* equals(double* output, const feature_node** x, const int l, const int n){
		int index;

		for(int i = 0; i < l*n; i++)
			output[i] = 0;

		for(int i = 0; i < l; i++){
			index = 0;
			while(x[i][index].index != -1){
				output[n*i + x[i][index].index-1] = x[i][index].value;
				index++;
			}
		}

		return output;
	}

	static feature_node** equals(feature_node** output, const double *x, const int l, const int n){
		int elements = 0;
		for(int i = 0; i < l*n; i++)
			if(x[i] != 0)
				elements++;

		feature_node* matrix_node = (feature_node*)Malloc(feature_node, elements+l);
		int index = 0;
		for(int i = 0; i < l; i++){
			output[i] = &matrix_node[index];
			for(int j = 0; j < n; j++){
				if(x[i*n+j] != 0){
					matrix_node[index].index = j+1;
					matrix_node[index].value = x[i*n+j];
					index++;
				}
			}
			matrix_node[index].index = -1;
			index++;
		}

		return output;
	}
};
//TODO functions for our solver
static void solve_l2r_l1l2_svc(const problem *prob, double *w, double eps,
	double Cp, double Cn, int solver_type);
static int* random_sampling(int* sample_id, const int sample_size,
	const int arr_size){ /*"inside-out" variant of Fisher–Yates shuffle*/
	std::random_device rd;
	std::mt19937 gen(rd());
	int *b = Malloc(int, arr_size);
	int tmp;
	for(int i = 0; i < arr_size; i++)
		b[i] = i;

	for(int i = arr_size-1; i >= arr_size-sample_size; i--) {
		std::uniform_int_distribution<> dis(0, i);
		int j = dis(gen);
		tmp = b[j];
		b[j] = b[i];
		b[i] = tmp;
	}
	for(int i = 0; i < sample_size; i++)
		sample_id[i] = b[arr_size-sample_size+i];
	free(b);
	return sample_id;
}
int compare(const void *a, const void *b){
	double c = *(double*)a;
	double d = *(double*)b;
	if(c < d){return -1;}
	else if(c == d){return 0;}
	else return 1;
}
static double* truncated_RBF(double *Q, const feature_node* const * A,
	const feature_node* const * B, const int nA, const int nB,
	const double threshold, const double gamma, const bool auto_threshold, double* a_threshold){
	double d_thresh_sq = log(threshold)/-gamma;
	int truncate_count = 0, atn;
	double* Q_s;
	for(int i=0; i<nA; i++){
		for(int j=0; j<nB; j++){
			const feature_node* xa = A[i];
			const feature_node* xb = B[j];
			double sum = 0;
			while(xa->index != -1 && xb->index != -1){
				if(xa->index == xb->index){
					double d = xa->value - xb->value;
					sum += d*d;
					++xa;
					++xb;
				} else{
					if(xa->index > xb->index){
						sum += xb->value * xb->value;
						++xb;
					} else{
						sum += xa->value * xa->value;
						++xa;
					}
				}
			}
			while(xa->index != -1){
				sum += xa->value * xa->value;
				++xa;
			}
			while(xb->index != -1){
				sum += xb->value * xb->value;
				++xb;
			}
			if(!auto_threshold){
				if(sum <= d_thresh_sq){
					Q[nB*i + j] = exp(-gamma*sum);
				}else{
					Q[nB*i + j] = 0;
					truncate_count++;
				}
			}
			else
				Q[nB*i + j] = exp(-gamma*sum);
		}
	}
	if(auto_threshold){
		Q_s = (double*)malloc(nA*nB*sizeof(double));
		for(int i = 0; i < nA*nB; i++)
			Q_s[i] = Q[i];
		qsort((void*)Q_s, nA*nB, sizeof(double), compare);
		atn = (int)(threshold*nA*nB);
		a_threshold[0] = Q_s[atn];
		fprintf(stderr, "auto_threshold = %lf\n", Q_s[atn]);
		for(int i = 0; i < nA*nB; i++)
			if(Q[i] <= Q_s[atn]){
				Q[i] = 0;
				truncate_count++;
			}
		free(Q_s);
	}

	if(nA>1) {
		fprintf(stderr, "thresh = %lf, %.4f %% entries of Q are truncated.\n",
			threshold, (double)truncate_count*100/(double)(nA*nB));
		fprintf(stderr, "d_thresh = %.4f, gamma = %.2f\n", sqrt(d_thresh_sq)
			, gamma);
	}
	return Q;
}
static double* random_projection(double* projected_matrix,
	const double* target_matrix, const double *y, const int l, const int n,
	const double* rp_arr, const int m, const bool SKIP_BIAS){
	/* target_matrix = [a11,...,a1n, a21,...,a2n, ..., al1,..,aln] */
	for(int i = 0; i < l; i++){
		for(int j = 0; j < m; j++){
			double value = 0;
			for(int k = 0; k < n; k++){
				value += rp_arr[n*j + k] * target_matrix[n*i + k];
			}
			if(SKIP_BIAS){
				projected_matrix[(m+1)*i + j] = y[i] * value;
			} else{
				projected_matrix[m*i + j] = y[i] * value;
			}
		}
	}
	return projected_matrix;
}
static void solve_r_ls_svm_svc(const problem *prob, double *w, feature_node **SV,
	const parameter *param, double Cp, double Cn, int m1, int m2, int cim1, int cjm1, int ci, int cj, double* a_threshold)
{
	int l = prob->l;
	feature_node **x = prob->x;
	double *y = prob->y;
	
	double *Q_r = new double[l*m1]; //TODO make it stored in sparse form
	int *sample_id = Malloc(int, m1);

	feature_node **selected_x = new feature_node*[m1];

	//random sampling
	int *sample_id1 = Malloc(int, cim1);
	int *sample_id2 = Malloc(int, cjm1);
	sample_id1 = random_sampling(sample_id1, cim1, ci);
	sample_id2 = random_sampling(sample_id2, cjm1, cj);
	for(int i = 0; i < cim1; i++)
		sample_id[i] = sample_id1[i];
	for(int i = 0; i < cjm1; i++)
		sample_id[i+cim1] = ci+sample_id2[i];
	free(sample_id1);
	free(sample_id2);
	for(int i = 0; i < m1; i++){
		selected_x[i] = x[sample_id[i]];
	}
	for(int i = 0; i < m1; i++){
		SV[i] = x[sample_id[i]];
	}
	//bias is at the end of each x, and won't affect the kernel value
	fprintf(stderr, "before RBF_truncate\n");
	Q_r = truncated_RBF(Q_r, x, selected_x, l, m1, param->threshold, param->gamma, param->auto_threshold, a_threshold);
	
	delete [] selected_x;

	double *rp_arr = new double[m2*m1];
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0,1.0/m2); //TODO sparse rp?
	for(int i = 0; i < m2; i++){
		for(int j = 0; j < m1; j++){
			rp_arr[i*m1 + j] = distribution(generator);
		}
	}

	//tmp = (long long)l*(long long)(m2+1);
	double *Q_rr = new double[l*(m2+1)]; //append bias at the end
	fprintf(stderr, "before random projection\n");
	Q_rr = random_projection(Q_rr, Q_r, y, l, m1, rp_arr, m2, true);
	for(int i = 0; i < l; i++){
		Q_rr[(m2+1)*i + m2] = prob->bias;
	}
	double eps = param->eps;
	struct problem mysubprob;
	mysubprob.bias = 1;
	mysubprob.n = m2+1;
	mysubprob.l = l;
	feature_node** sparse_Q_rr = (feature_node**)Malloc(feature_node*, l);
	sparse_Q_rr = sparse_operator::equals(sparse_Q_rr, Q_rr, l, m2+1);
	mysubprob.x = sparse_Q_rr;
	mysubprob.y = prob->y;
	double *alpha = (double*)Malloc(double, m2+1);
	solve_l2r_l1l2_svc(&mysubprob, alpha, eps, Cp, Cn, L2R_L2LOSS_SVC_DUAL);
//	fprintf(stderr, "after solve_l2r_l1l2_svc\n");
	w[m1] = alpha[m2]; //weight for bias, w should be of length m1+1
	for(int i = 0; i < m1; i++){
		w[i] = 0;
		for(int j = 0; j < m2; j++){
			w[i] += rp_arr[m1*j + i] * alpha[j];
		}
		w[i] = w[i]*y[sample_id[i]];
	}
	free(sample_id);
	free(alpha);
	delete [] Q_r;
	delete [] Q_rr;
	delete [] rp_arr;
}
static void solve_trsvm_svc(const problem *prob, double *w, feature_node **SV,
	const parameter *param, double Cp, double Cn, int m1, int cim1, int cjm1, int ci, int cj, double* a_threshold)
{
	int l = prob->l;
	feature_node **x = prob->x;
	
	double *Q_r = new double[l*m1]; //TODO make it stored in sparse form
	int *sample_id = Malloc(int, m1);

	feature_node **selected_x = new feature_node*[m1];

	//random sampling
	int *sample_id1 = Malloc(int, cim1);
	int *sample_id2 = Malloc(int, cjm1);
	sample_id1 = random_sampling(sample_id1, cim1, ci);
	sample_id2 = random_sampling(sample_id2, cjm1, cj);
	for(int i = 0; i < cim1; i++)
		sample_id[i] = sample_id1[i];
	for(int i = 0; i < cjm1; i++)
		sample_id[i+cim1] = ci+sample_id2[i];
	free(sample_id1);
	free(sample_id2);
	for(int i = 0; i < m1; i++){
		selected_x[i] = x[sample_id[i]];
	}
	for(int i = 0; i < m1; i++){
		SV[i] = x[sample_id[i]];
	}
	//bias is at the end of each x, and won't affect the kernel value
	fprintf(stderr, "before RBF_truncate\n");
	Q_r = truncated_RBF(Q_r, x, selected_x, l, m1, param->threshold, param->gamma, param->auto_threshold, a_threshold);
	fprintf(stderr, "after RBF_truncate\n");
	
	delete [] selected_x;

	double *Q_rr = new double[l*(m1+1)]; //append bias at the end
	
	for(int i = 0; i < l; i++){
		for(int j = 0; j < m1; j++)
			Q_rr[(m1+1)*i + j] = Q_r[i*m1+j];
		Q_rr[(m1+1)*i + m1] = prob->bias;
	}
	double eps = param->eps;
	struct problem mysubprob;
	mysubprob.bias = 1;
	mysubprob.n = m1+1;
	mysubprob.l = l;
	feature_node** sparse_Q_rr = (feature_node**)Malloc(feature_node*, l);
	sparse_Q_rr = sparse_operator::equals(sparse_Q_rr, Q_rr, l, m1+1);
	mysubprob.x = sparse_Q_rr;
	mysubprob.y = prob->y;
//	double *alpha = (double*)Malloc(double, m1+1);
	fprintf(stderr, "before before_solver\n");
	solve_l2r_l1l2_svc(&mysubprob, w, eps, Cp, Cn, L2R_L2LOSS_SVC_DUAL);
//	fprintf(stderr, "after solve_l2r_l1l2_svc\n");
	delete [] Q_r;
	delete [] Q_rr;
}

class l2r_lr_fun: public function
{
public:
	l2r_lr_fun(const problem *prob, double *C);
	~l2r_lr_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	const problem *prob;
};

l2r_lr_fun::l2r_lr_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	this->C = C;
}

l2r_lr_fun::~l2r_lr_fun()
{
	delete[] z;
	delete[] D;
}


double l2r_lr_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{
		double yz = y[i]*z[i];
		if (yz >= 0)
			f += C[i]*log(1 + exp(-yz));
		else
			f += C[i]*(-yz+log(1 + exp(yz)));
	}

	return(f);
}

void l2r_lr_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + g[i];
}

int l2r_lr_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	double *wa = new double[l];
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i] = 0;
	for(i=0;i<l;i++)
	{
		feature_node * const xi=x[i];
		wa[i] = sparse_operator::dot(s, xi);

		wa[i] = C[i]*D[i]*wa[i];

		sparse_operator::axpy(wa[i], xi, Hs);
	}
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + Hs[i];
	delete[] wa;
}

void l2r_lr_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=sparse_operator::dot(v, x[i]);
}

void l2r_lr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
		sparse_operator::axpy(v[i], x[i], XTv);
}

class l2r_l2_svc_fun: public function
{
public:
	l2r_l2_svc_fun(const problem *prob, double *C);
	~l2r_l2_svc_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

protected:
	void Xv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	int *I;
	int sizeI;
	const problem *prob;
};

l2r_l2_svc_fun::l2r_l2_svc_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	I = new int[l];
	this->C = C;
}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	delete[] z;
	delete[] D;
	delete[] I;
}

double l2r_l2_svc_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += C[i]*d*d;
	}

	return(f);
}

void l2r_l2_svc_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

int l2r_l2_svc_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int w_size=get_nr_variable();
	double *wa = new double[sizeI];
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node * const xi=x[I[i]];
		wa[i] = sparse_operator::dot(s, xi);

		wa[i] = C[I[i]]*wa[i];

		sparse_operator::axpy(wa[i], xi, Hs);
	}
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2r_l2_svc_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=sparse_operator::dot(v, x[i]);
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
		sparse_operator::axpy(v[i], x[I[i]], XTv);
}

class l2r_l2_svr_fun: public l2r_l2_svc_fun
{
public:
	l2r_l2_svr_fun(const problem *prob, double *C, double p);

	double fun(double *w);
	void grad(double *w, double *g);

private:
	double p;
};

l2r_l2_svr_fun::l2r_l2_svr_fun(const problem *prob, double *C, double p):
	l2r_l2_svc_fun(prob, C)
{
	this->p = p;
}

double l2r_l2_svr_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2;
	for(i=0;i<l;i++)
	{
		d = z[i] - y[i];
		if(d < -p)
			f += C[i]*(d+p)*(d+p);
		else if(d > p)
			f += C[i]*(d-p)*(d-p);
	}

	return(f);
}

void l2r_l2_svr_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;

	sizeI = 0;
	for(i=0;i<l;i++)
	{
		d = z[i] - y[i];

		// generate index set I
		if(d < -p)
		{
			z[sizeI] = C[i]*(d+p);
			I[sizeI] = i;
			sizeI++;
		}
		else if(d > p)
		{
			z[sizeI] = C[i]*(d-p);
			I[sizeI] = i;
			sizeI++;
		}

	}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

// A coordinate descent algorithm for
// multi-class support vector machines by Crammer and Singer
//
//  min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
//    s.t.     \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
//
//  where e^m_i = 0 if y_i  = m,
//        e^m_i = 1 if y_i != m,
//  C^m_i = C if m  = y_i,
//  C^m_i = 0 if m != y_i,
//  and w_m(\alpha) = \sum_i \alpha^m_i x_i
//
// Given:
// x, y, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Appendix of LIBLINEAR paper, Fan et al. (2008)

#define GETI(i) ((int) prob->y[i])
// To support weights for instances, use GETI(i) (i)

class Solver_MCSVM_CS
{
	public:
		Solver_MCSVM_CS(const problem *prob, int nr_class, double *C, double eps=0.1, int max_iter=100000);
		~Solver_MCSVM_CS();
		void Solve(double *w);
	private:
		void solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new);
		bool be_shrunk(int i, int m, int yi, double alpha_i, double minG);
		double *B, *C, *G;
		int w_size, l;
		int nr_class;
		int max_iter;
		double eps;
		const problem *prob;
};

Solver_MCSVM_CS::Solver_MCSVM_CS(const problem *prob, int nr_class, double *weighted_C, double eps, int max_iter)
{
	this->w_size = prob->n;
	this->l = prob->l;
	this->nr_class = nr_class;
	this->eps = eps;
	this->max_iter = max_iter;
	this->prob = prob;
	this->B = new double[nr_class];
	this->G = new double[nr_class];
	this->C = weighted_C;
}

Solver_MCSVM_CS::~Solver_MCSVM_CS()
{
	delete[] B;
	delete[] G;
}

int compare_double(const void *a, const void *b)
{
	if(*(double *)a > *(double *)b)
		return -1;
	if(*(double *)a < *(double *)b)
		return 1;
	return 0;
}

void Solver_MCSVM_CS::solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new)
{
	int r;
	double *D;

	clone(D, B, active_i);
	if(yi < active_i)
		D[yi] += A_i*C_yi;
	qsort(D, active_i, sizeof(double), compare_double);

	double beta = D[0] - A_i*C_yi;
	for(r=1;r<active_i && beta<r*D[r];r++)
		beta += D[r];
	beta /= r;

	for(r=0;r<active_i;r++)
	{
		if(r == yi)
			alpha_new[r] = min(C_yi, (beta-B[r])/A_i);
		else
			alpha_new[r] = min((double)0, (beta - B[r])/A_i);
	}
	delete[] D;
}

bool Solver_MCSVM_CS::be_shrunk(int i, int m, int yi, double alpha_i, double minG)
{
	double bound = 0;
	if(m == yi)
		bound = C[GETI(i)];
	if(alpha_i == bound && G[m] < minG)
		return true;
	return false;
}

void Solver_MCSVM_CS::Solve(double *w)
{
	int i, m, s;
	int iter = 0;
	double *alpha =  new double[l*nr_class];
	double *alpha_new = new double[nr_class];
	int *index = new int[l];
	double *QD = new double[l];
	int *d_ind = new int[nr_class];
	double *d_val = new double[nr_class];
	int *alpha_index = new int[nr_class*l];
	int *y_index = new int[l];
	int active_size = l;
	int *active_size_i = new int[l];
	double eps_shrink = max(10.0*eps, 1.0); // stopping tolerance for shrinking
	bool start_from_all = true;

	// Initial alpha can be set here. Note that
	// sum_m alpha[i*nr_class+m] = 0, for all i=1,...,l-1
	// alpha[i*nr_class+m] <= C[GETI(i)] if prob->y[i] == m
	// alpha[i*nr_class+m] <= 0 if prob->y[i] != m
	// If initial alpha isn't zero, uncomment the for loop below to initialize w
	for(i=0;i<l*nr_class;i++)
		alpha[i] = 0;

	for(i=0;i<w_size*nr_class;i++)
		w[i] = 0;
	for(i=0;i<l;i++)
	{
		for(m=0;m<nr_class;m++)
			alpha_index[i*nr_class+m] = m;
		feature_node *xi = prob->x[i];
		QD[i] = 0;
		while(xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;

			// Uncomment the for loop if initial alpha isn't zero
			// for(m=0; m<nr_class; m++)
			//	w[(xi->index-1)*nr_class+m] += alpha[i*nr_class+m]*val;
			xi++;
		}
		active_size_i[i] = nr_class;
		y_index[i] = (int)prob->y[i];
		index[i] = i;
	}

	while(iter < max_iter)
	{
		double stopping = -INF;
		for(i=0;i<active_size;i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}
		for(s=0;s<active_size;s++)
		{
			i = index[s];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*nr_class];
			int *alpha_index_i = &alpha_index[i*nr_class];

			if(Ai > 0)
			{
				for(m=0;m<active_size_i[i];m++)
					G[m] = 1;
				if(y_index[i] < active_size_i[i])
					G[y_index[i]] = 0;

				feature_node *xi = prob->x[i];
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<active_size_i[i];m++)
						G[m] += w_i[alpha_index_i[m]]*(xi->value);
					xi++;
				}

				double minG = INF;
				double maxG = -INF;
				for(m=0;m<active_size_i[i];m++)
				{
					if(alpha_i[alpha_index_i[m]] < 0 && G[m] < minG)
						minG = G[m];
					if(G[m] > maxG)
						maxG = G[m];
				}
				if(y_index[i] < active_size_i[i])
					if(alpha_i[(int) prob->y[i]] < C[GETI(i)] && G[y_index[i]] < minG)
						minG = G[y_index[i]];

				for(m=0;m<active_size_i[i];m++)
				{
					if(be_shrunk(i, m, y_index[i], alpha_i[alpha_index_i[m]], minG))
					{
						active_size_i[i]--;
						while(active_size_i[i]>m)
						{
							if(!be_shrunk(i, active_size_i[i], y_index[i],
											alpha_i[alpha_index_i[active_size_i[i]]], minG))
							{
								swap(alpha_index_i[m], alpha_index_i[active_size_i[i]]);
								swap(G[m], G[active_size_i[i]]);
								if(y_index[i] == active_size_i[i])
									y_index[i] = m;
								else if(y_index[i] == m)
									y_index[i] = active_size_i[i];
								break;
							}
							active_size_i[i]--;
						}
					}
				}

				if(active_size_i[i] <= 1)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}

				if(maxG-minG <= 1e-12)
					continue;
				else
					stopping = max(maxG - minG, stopping);

				for(m=0;m<active_size_i[i];m++)
					B[m] = G[m] - Ai*alpha_i[alpha_index_i[m]] ;

				solve_sub_problem(Ai, y_index[i], C[GETI(i)], active_size_i[i], alpha_new);
				int nz_d = 0;
				for(m=0;m<active_size_i[i];m++)
				{
					double d = alpha_new[m] - alpha_i[alpha_index_i[m]];
					alpha_i[alpha_index_i[m]] = alpha_new[m];
					if(fabs(d) >= 1e-12)
					{
						d_ind[nz_d] = alpha_index_i[m];
						d_val[nz_d] = d;
						nz_d++;
					}
				}

				xi = prob->x[i];
				while(xi->index != -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<nz_d;m++)
						w_i[d_ind[m]] += d_val[m]*xi->value;
					xi++;
				}
			}
		}

		iter++;
		if(iter % 10 == 0)
		{
			info(".");
		}

		if(stopping < eps_shrink)
		{
			if(stopping < eps && start_from_all == true)
				break;
			else
			{
				active_size = l;
				for(i=0;i<l;i++)
					active_size_i[i] = nr_class;
				info("*");
				eps_shrink = max(eps_shrink/2, eps);
				start_from_all = true;
			}
		}
		else
			start_from_all = false;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0;i<w_size*nr_class;i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0;i<l*nr_class;i++)
	{
		v += alpha[i];
		if(fabs(alpha[i]) > 0)
			nSV++;
	}
	for(i=0;i<l;i++)
		v -= alpha[i*nr_class+(int)prob->y[i]];
	info("Objective value = %lf\n",v);
	info("nSV = %d\n",nSV);

	delete [] alpha;
	delete [] alpha_new;
	delete [] index;
	delete [] QD;
	delete [] d_ind;
	delete [] d_val;
	delete [] alpha_index;
	delete [] y_index;
	delete [] active_size_i;
}

// A coordinate descent algorithm for
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= \alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
// 		upper_bound_i = Cp if y_i = 1
// 		upper_bound_i = Cn if y_i = -1
// 		D_ii = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		D_ii = 1/(2*Cp)	if y_i = 1
// 		D_ii = 1/(2*Cn)	if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 3 of Hsieh et al., ICML 2008

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svc(
	const problem *prob, double *w, double eps,
	double Cp, double Cn, int solver_type)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 10000;
	int *index = new int[l];
	double *alpha = new double[l];
	schar *y = new schar[l];
	int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
	}

	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]
	for(i=0; i<l; i++)
		alpha[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	fprintf(stderr, "In solve_l2r_l1l2_svc, w_size = m2+1 = %d, l = %d\n", w_size, l);
	for(i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];

		feature_node * const xi = prob->x[i];
		QD[i] += sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(y[i]*alpha[i], xi, w);

		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			i = index[s];
			const schar yi = y[i];
			feature_node * const xi = prob->x[i];

			G = yi*sparse_operator::dot(w, xi)-1;

			C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;
				sparse_operator::axpy(d, xi, w);
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	info("nSV = %d\n",nSV);

	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
}


// A coordinate descent algorithm for
// L1-loss and L2-loss epsilon-SVR dual problem
//
//  min_\beta  0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| + \sum_{i=1}^l yi\beta_i,
//    s.t.      -upper_bound_i <= \beta_i <= upper_bound_i,
//
//  where Qij = xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
// 		upper_bound_i = C
// 		lambda_i = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		lambda_i = 1/(2*C)
//
// Given:
// x, y, p, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 4 of Ho and Lin, 2012

#undef GETI
#define GETI(i) (0)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svr(
	const problem *prob, double *w, const parameter *param,
	int solver_type)
{
	int l = prob->l;
	double C = param->C;
	double p = param->p;
	int w_size = prob->n;
	double eps = param->eps;
	int i, s, iter = 0;
	int max_iter = 1000;
	int active_size = l;
	int *index = new int[l];

	double d, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double *beta = new double[l];
	double *QD = new double[l];
	double *y = prob->y;

	// L2R_L2LOSS_SVR_DUAL
	double lambda[1], upper_bound[1];
	lambda[0] = 0.5/C;
	upper_bound[0] = INF;

	if(solver_type == L2R_L1LOSS_SVR_DUAL)
	{
		lambda[0] = 0;
		upper_bound[0] = C;
	}

	// Initial beta can be set here. Note that
	// -upper_bound <= beta[i] <= upper_bound
	for(i=0; i<l; i++)
		beta[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node * const xi = prob->x[i];
		QD[i] = sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(beta[i], xi, w);

		index[i] = i;
	}


	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			G = -y[i] + lambda[GETI(i)]*beta[i];
			H = QD[i] + lambda[GETI(i)];

			feature_node * const xi = prob->x[i];
			G += sparse_operator::dot(w, xi);

			double Gp = G+p;
			double Gn = G-p;
			double violation = 0;
			if(beta[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old && Gn<-Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] >= upper_bound[GETI(i)])
			{
				if(Gp > 0)
					violation = Gp;
				else if(Gp < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] <= -upper_bound[GETI(i)])
			{
				if(Gn < 0)
					violation = -Gn;
				else if(Gn > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*beta[i])
				d = -Gp/H;
			else if(Gn > H*beta[i])
				d = -Gn/H;
			else
				d = -beta[i];

			if(fabs(d) < 1.0e-12)
				continue;

			double beta_old = beta[i];
			beta[i] = min(max(beta[i]+d, -upper_bound[GETI(i)]), upper_bound[GETI(i)]);
			d = beta[i]-beta_old;

			if(d != 0)
				sparse_operator::axpy(d, xi, w);
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda[GETI(i)]*beta[i]*beta[i];
		if(beta[i] != 0)
			nSV++;
	}

	info("Objective value = %lf\n", v);
	info("nSV = %d\n",nSV);

	delete [] beta;
	delete [] QD;
	delete [] index;
}


// A coordinate descent algorithm for
// the dual of L2-regularized logistic regression problems
//
//  min_\alpha  0.5(\alpha^T Q \alpha) + \sum \alpha_i log (\alpha_i) + (upper_bound_i - \alpha_i) log (upper_bound_i - \alpha_i),
//    s.t.      0 <= \alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  upper_bound_i = Cp if y_i = 1
//  upper_bound_i = Cn if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 5 of Yu et al., MLJ 2010

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

void solve_l2r_lr_dual(const problem *prob, double *w, double eps, double Cp, double Cn)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double *xTx = new double[l];
	int max_iter = 1000;
	int *index = new int[l];
	double *alpha = new double[2*l]; // store alpha and C - alpha
	schar *y = new schar[l];
	int max_inner_iter = 100; // for inner Newton
	double innereps = 1e-2;
	double innereps_min = min(1e-8, eps);
	double upper_bound[3] = {Cn, 0, Cp};

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
	}

	// Initial alpha can be set here. Note that
	// 0 < alpha[i] < upper_bound[GETI(i)]
	// alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
	for(i=0; i<l; i++)
	{
		alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
		alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];
	}

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node * const xi = prob->x[i];
		xTx[i] = sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(y[i]*alpha[2*i], xi, w);
		index[i] = i;
	}

	while (iter < max_iter)
	{
		for (i=0; i<l; i++)
		{
			int j = i+rand()%(l-i);
			swap(index[i], index[j]);
		}
		int newton_iter = 0;
		double Gmax = 0;
		for (s=0; s<l; s++)
		{
			i = index[s];
			const schar yi = y[i];
			double C = upper_bound[GETI(i)];
			double ywTx = 0, xisq = xTx[i];
			feature_node * const xi = prob->x[i];
			ywTx = yi*sparse_operator::dot(w, xi);
			double a = xisq, b = ywTx;

			// Decide to minimize g_1(z) or g_2(z)
			int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
			if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
			{
				ind1 = 2*i+1;
				ind2 = 2*i;
				sign = -1;
			}

			//  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
			double alpha_old = alpha[ind1];
			double z = alpha_old;
			if(C - z < 0.5 * C)
				z = 0.1*z;
			double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
			Gmax = max(Gmax, fabs(gp));

			// Newton method on the sub-problem
			const double eta = 0.1; // xi in the paper
			int inner_iter = 0;
			while (inner_iter <= max_inner_iter)
			{
				if(fabs(gp) < innereps)
					break;
				double gpp = a + C/(C-z)/z;
				double tmpz = z - gp/gpp;
				if(tmpz <= 0)
					z *= eta;
				else // tmpz in (0, C)
					z = tmpz;
				gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
				newton_iter++;
				inner_iter++;
			}

			if(inner_iter > 0) // update w
			{
				alpha[ind1] = z;
				alpha[ind2] = C-z;
				sparse_operator::axpy(sign*(z-alpha_old)*yi, xi, w);
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gmax < eps)
			break;

		if(newton_iter <= l/10)
			innereps = max(innereps_min, 0.1*innereps);

	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 0 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	for(i=0; i<w_size; i++)
		v += w[i] * w[i];
	v *= 0.5;
	for(i=0; i<l; i++)
		v += alpha[2*i] * log(alpha[2*i]) + alpha[2*i+1] * log(alpha[2*i+1])
			- upper_bound[GETI(i)] * log(upper_bound[GETI(i)]);
	info("Objective value = %lf\n", v);

	delete [] xTx;
	delete [] alpha;
	delete [] y;
	delete [] index;
}

// A coordinate descent algorithm for
// L1-regularized L2-loss support vector classification
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2010) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_l2_svc(
	problem *prob_col, double *w, double eps,
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, iter = 0;
	int max_iter = 1000;
	int active_size = w_size;
	int max_num_linesearch = 20;

	double sigma = 0.01;
	double d, G_loss, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double d_old, d_diff;
	double loss_old, loss_new;
	double appxcond, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *b = new double[l]; // b = 1-ywTx
	double *xj_sq = new double[w_size];
	feature_node *x;

	double C[3] = {Cn,0,Cp};

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		b[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;
	}
	for(j=0; j<w_size; j++)
	{
		index[j] = j;
		xj_sq[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x->value *= y[ind]; // x->value stores yi*xij
			double val = x->value;
			b[ind] -= w[j]*val;
			xj_sq[j] += C[GETI(ind)]*val*val;
			x++;
		}
	}

	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(j=0; j<active_size; j++)
		{
			int i = j+rand()%(active_size-j);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			G_loss = 0;
			H = 0;

			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				if(b[ind] > 0)
				{
					double val = x->value;
					double tmp = C[GETI(ind)]*val;
					G_loss -= tmp*b[ind];
					H += tmp*val;
				}
				x++;
			}
			G_loss *= 2;

			G = G_loss;
			H *= 2;
			H = max(H, 1e-12);

			double Gp = G+1;
			double Gn = G-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*w[j])
				d = -Gp/H;
			else if(Gn > H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < 1.0e-12)
				continue;

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			d_old = 0;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				d_diff = d_old - d;
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

				appxcond = xj_sq[j]*d*d + G_loss*d + cond;
				if(appxcond <= 0)
				{
					x = prob_col->x[j];
					sparse_operator::axpy(d_diff, x, b);
					break;
				}

				if(num_linesearch == 0)
				{
					loss_old = 0;
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						if(b[ind] > 0)
							loss_old += C[GETI(ind)]*b[ind]*b[ind];
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}
				else
				{
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}

				cond = cond + loss_new - loss_old;
				if(cond <= 0)
					break;
				else
				{
					d_old = d;
					d *= 0.5;
					delta *= 0.5;
				}
			}

			w[j] += d;

			// recompute b[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				info("#");
				for(int i=0; i<l; i++)
					b[i] = 1;

				for(int i=0; i<w_size; i++)
				{
					if(w[i]==0) continue;
					x = prob_col->x[i];
					sparse_operator::axpy(-w[i], x, b);
				}
			}
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
	{
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= prob_col->y[x->index-1]; // restore x->value
			x++;
		}
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	}
	for(j=0; j<l; j++)
		if(b[j] > 0)
			v += C[GETI(j)]*b[j]*b[j];

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] b;
	delete [] xj_sq;
}

// A coordinate descent algorithm for
// L1-regularized logistic regression problems
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2011) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_lr(
	const problem *prob_col, double *w, double eps,
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, newton_iter=0, iter=0;
	int max_newton_iter = 100;
	int max_iter = 1000;
	int max_num_linesearch = 20;
	int active_size;
	int QP_active_size;

	double nu = 1e-12;
	double inner_eps = 1;
	double sigma = 0.01;
	double w_norm, w_norm_new;
	double z, G, H;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double QP_Gmax_old = INF;
	double QP_Gmax_new, QP_Gnorm1_new;
	double delta, negsum_xTd, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *Hdiag = new double[w_size];
	double *Grad = new double[w_size];
	double *wpd = new double[w_size];
	double *xjneg_sum = new double[w_size];
	double *xTd = new double[l];
	double *exp_wTx = new double[l];
	double *exp_wTx_new = new double[l];
	double *tau = new double[l];
	double *D = new double[l];
	feature_node *x;

	double C[3] = {Cn,0,Cp};

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;

		exp_wTx[j] = 0;
	}

	w_norm = 0;
	for(j=0; j<w_size; j++)
	{
		w_norm += fabs(w[j]);
		wpd[j] = w[j];
		index[j] = j;
		xjneg_sum[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			double val = x->value;
			exp_wTx[ind] += w[j]*val;
			if(y[ind] == -1)
				xjneg_sum[j] += C[GETI(ind)]*val;
			x++;
		}
	}
	for(j=0; j<l; j++)
	{
		exp_wTx[j] = exp(exp_wTx[j]);
		double tau_tmp = 1/(1+exp_wTx[j]);
		tau[j] = C[GETI(j)]*tau_tmp;
		D[j] = C[GETI(j)]*exp_wTx[j]*tau_tmp*tau_tmp;
	}

	while(newton_iter < max_newton_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;
		active_size = w_size;

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			Hdiag[j] = nu;
			Grad[j] = 0;

			double tmp = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				Hdiag[j] += x->value*x->value*D[ind];
				tmp += x->value*tau[ind];
				x++;
			}
			Grad[j] = -tmp + xjneg_sum[j];

			double Gp = Grad[j]+1;
			double Gn = Grad[j]-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				//outer-level shrinking
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;
		}

		if(newton_iter == 0)
			Gnorm1_init = Gnorm1_new;

		if(Gnorm1_new <= eps*Gnorm1_init)
			break;

		iter = 0;
		QP_Gmax_old = INF;
		QP_active_size = active_size;

		for(int i=0; i<l; i++)
			xTd[i] = 0;

		// optimize QP over wpd
		while(iter < max_iter)
		{
			QP_Gmax_new = 0;
			QP_Gnorm1_new = 0;

			for(j=0; j<QP_active_size; j++)
			{
				int i = j+rand()%(QP_active_size-j);
				swap(index[i], index[j]);
			}

			for(s=0; s<QP_active_size; s++)
			{
				j = index[s];
				H = Hdiag[j];

				x = prob_col->x[j];
				G = Grad[j] + (wpd[j]-w[j])*nu;
				while(x->index != -1)
				{
					int ind = x->index-1;
					G += x->value*D[ind]*xTd[ind];
					x++;
				}

				double Gp = G+1;
				double Gn = G-1;
				double violation = 0;
				if(wpd[j] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					//inner-level shrinking
					else if(Gp>QP_Gmax_old/l && Gn<-QP_Gmax_old/l)
					{
						QP_active_size--;
						swap(index[s], index[QP_active_size]);
						s--;
						continue;
					}
				}
				else if(wpd[j] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				QP_Gmax_new = max(QP_Gmax_new, violation);
				QP_Gnorm1_new += violation;

				// obtain solution of one-variable problem
				if(Gp < H*wpd[j])
					z = -Gp/H;
				else if(Gn > H*wpd[j])
					z = -Gn/H;
				else
					z = -wpd[j];

				if(fabs(z) < 1.0e-12)
					continue;
				z = min(max(z,-10.0),10.0);

				wpd[j] += z;

				x = prob_col->x[j];
				sparse_operator::axpy(z, x, xTd);
			}

			iter++;

			if(QP_Gnorm1_new <= inner_eps*Gnorm1_init)
			{
				//inner stopping
				if(QP_active_size == active_size)
					break;
				//active set reactivation
				else
				{
					QP_active_size = active_size;
					QP_Gmax_old = INF;
					continue;
				}
			}

			QP_Gmax_old = QP_Gmax_new;
		}

		if(iter >= max_iter)
			info("WARNING: reaching max number of inner iterations\n");

		delta = 0;
		w_norm_new = 0;
		for(j=0; j<w_size; j++)
		{
			delta += Grad[j]*(wpd[j]-w[j]);
			if(wpd[j] != 0)
				w_norm_new += fabs(wpd[j]);
		}
		delta += (w_norm_new-w_norm);

		negsum_xTd = 0;
		for(int i=0; i<l; i++)
			if(y[i] == -1)
				negsum_xTd += C[GETI(i)]*xTd[i];

		int num_linesearch;
		for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
		{
			cond = w_norm_new - w_norm + negsum_xTd - sigma*delta;

			for(int i=0; i<l; i++)
			{
				double exp_xTd = exp(xTd[i]);
				exp_wTx_new[i] = exp_wTx[i]*exp_xTd;
				cond += C[GETI(i)]*log((1+exp_wTx_new[i])/(exp_xTd+exp_wTx_new[i]));
			}

			if(cond <= 0)
			{
				w_norm = w_norm_new;
				for(j=0; j<w_size; j++)
					w[j] = wpd[j];
				for(int i=0; i<l; i++)
				{
					exp_wTx[i] = exp_wTx_new[i];
					double tau_tmp = 1/(1+exp_wTx[i]);
					tau[i] = C[GETI(i)]*tau_tmp;
					D[i] = C[GETI(i)]*exp_wTx[i]*tau_tmp*tau_tmp;
				}
				break;
			}
			else
			{
				w_norm_new = 0;
				for(j=0; j<w_size; j++)
				{
					wpd[j] = (w[j]+wpd[j])*0.5;
					if(wpd[j] != 0)
						w_norm_new += fabs(wpd[j]);
				}
				delta *= 0.5;
				negsum_xTd *= 0.5;
				for(int i=0; i<l; i++)
					xTd[i] *= 0.5;
			}
		}

		// Recompute some info due to too many line search steps
		if(num_linesearch >= max_num_linesearch)
		{
			for(int i=0; i<l; i++)
				exp_wTx[i] = 0;

			for(int i=0; i<w_size; i++)
			{
				if(w[i]==0) continue;
				x = prob_col->x[i];
				sparse_operator::axpy(w[i], x, exp_wTx);
			}

			for(int i=0; i<l; i++)
				exp_wTx[i] = exp(exp_wTx[i]);
		}

		if(iter == 1)
			inner_eps *= 0.25;

		newton_iter++;
		Gmax_old = Gmax_new;

		info("iter %3d  #CD cycles %d\n", newton_iter, iter);
	}

	info("=========================\n");
	info("optimization finished, #iter = %d\n", newton_iter);
	if(newton_iter >= max_newton_iter)
		info("WARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	for(j=0; j<l; j++)
		if(y[j] == 1)
			v += C[GETI(j)]*log(1+1/exp_wTx[j]);
		else
			v += C[GETI(j)]*log(1+exp_wTx[j]);

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] Hdiag;
	delete [] Grad;
	delete [] wpd;
	delete [] xjneg_sum;
	delete [] xTd;
	delete [] exp_wTx;
	delete [] exp_wTx_new;
	delete [] tau;
	delete [] D;
}

// transpose matrix X from row format to column format
static void transpose(const problem *prob, feature_node **x_space_ret, problem *prob_col)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	size_t nnz = 0;
	size_t *col_ptr = new size_t [n+1];
	feature_node *x_space;
	prob_col->l = l;
	prob_col->n = n;
	prob_col->y = new double[l];
	prob_col->x = new feature_node*[n];

	for(i=0; i<l; i++)
		prob_col->y[i] = prob->y[i];

	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			nnz++;
			col_ptr[x->index]++;
			x++;
		}
	}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;

	x_space = new feature_node[nnz+n];
	for(i=0; i<n; i++)
		prob_col->x[i] = &x_space[col_ptr[i]];

	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x_space[col_ptr[ind]].index = i+1; // starts from 1
			x_space[col_ptr[ind]].value = x->value;
			col_ptr[ind]++;
			x++;
		}
	}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;

	*x_space_ret = x_space;

	delete [] col_ptr;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set.
	// However, for two-class sets with -1/+1 labels and -1 appears first,
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
	{
		swap(label[0],label[1]);
		swap(count[0],count[1]);
		for(i=0;i<l;i++)
		{
			if(data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn)
{
	//inner and outer tolerances for TRON
	double eps = param->eps;
	double eps_cg = 0.1;
	if(param->init_sol != NULL)
		eps_cg = 0.5;

	int pos = 0;
	int neg = 0;
	for(int i=0;i<prob->l;i++)
		if(prob->y[i] > 0)
			pos++;
	neg = prob->l - pos;
	double primal_solver_tol = eps*max(min(pos,neg), 1)/prob->l;

	function *fun_obj=NULL;
	switch(param->solver_type)
	{
		case L2R_LR:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_lr_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L2R_L2LOSS_SVC:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_l2_svc_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L2R_L2LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L2LOSS_SVC_DUAL);
			break;
		case L2R_L1LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
			break;
		case L1R_L2LOSS_SVC:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_l2_svc(&prob_col, w, primal_solver_tol, Cp, Cn);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case L1R_LR:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_lr(&prob_col, w, primal_solver_tol, Cp, Cn);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case L2R_LR_DUAL:
			solve_l2r_lr_dual(prob, w, eps, Cp, Cn);
			break;
		case L2R_L2LOSS_SVR:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
				C[i] = param->C;

			fun_obj=new l2r_l2_svr_fun(prob, C, param->p);
			TRON tron_obj(fun_obj, param->eps);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;

		}
		case L2R_L1LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(prob, w, param, L2R_L1LOSS_SVR_DUAL);
			break;
		case L2R_L2LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(prob, w, param, L2R_L2LOSS_SVR_DUAL);
			break;
		case R_LS_SVM:
			fprintf(stderr, "Going to solve_r_ls_svm_svc");
//			solve_r_ls_svm_svc(prob, w, param, Cp, Cn);
			break;
		default:
			fprintf(stderr, "ERROR: unknown solver_type\n");
			break;
	}
}

// Calculate the initial C for parameter selection
static double calc_start_C(const problem *prob, const parameter *param)
{
	int i;
	double xTx,max_xTx;
	max_xTx = 0;
	for(i=0; i<prob->l; i++)
	{
		xTx = 0;
		feature_node *xi=prob->x[i];
		while(xi->index != -1)
		{
			double val = xi->value;
			xTx += val*val;
			xi++;
		}
		if(xTx > max_xTx)
			max_xTx = xTx;
	}

	double min_C = 1.0;
	if(param->solver_type == L2R_LR)
		min_C = 1.0 / (prob->l * max_xTx);
	else if(param->solver_type == L2R_L2LOSS_SVC)
		min_C = 1.0 / (2 * prob->l * max_xTx);

	return pow( 2, floor(log(min_C) / log(2.0)) );
}


//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i,j;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;

	model *model_ = Malloc(model,1);

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	if(check_regression_model(model_))
	{
		model_->w = Malloc(double, w_size);
		for(i=0; i<w_size; i++)
			model_->w[i] = 0;
		model_->nr_class = 2;
		model_->label = NULL;
		train_one(prob, param, model_->w, 0, 0);
	}
	else
	{
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		group_classes(prob,&nr_class,&label,&start,&count,perm);

		model_->nr_class=nr_class;
		model_->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model_->label[i] = label[i];

		// calculate weighted C
		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// constructing the subproblem
		feature_node **x = Malloc(feature_node *,l);
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];


		// multi-class svm by Crammer and Singer
		if(param->solver_type == MCSVM_CS)
		{
			int k;
			problem sub_prob;
			sub_prob.l = l;
			sub_prob.n = n;
			sub_prob.x = Malloc(feature_node *,sub_prob.l);
			sub_prob.y = Malloc(double, sub_prob.l);

			for(k=0; k<sub_prob.l; k++)
				sub_prob.x[k] = x[k];

			model_->w=Malloc(double, n*nr_class);
			for(i=0;i<nr_class;i++)
				for(j=start[i];j<start[i]+count[i];j++)
					sub_prob.y[j] = i;
			Solver_MCSVM_CS Solver(&sub_prob, nr_class, weighted_C, param->eps);
			Solver.Solve(model_->w);
			free(sub_prob.x);
			free(sub_prob.y);
		}
		else
		{
			if(param->solver_type == R_LS_SVM){
				fprintf(stderr, "In R_LS_SVM, class_nr = %d\n", nr_class);
				double r_size_d, rp_size_d;
				int rp_size_i;
				if(param->auto_threshold)
					model_->a_threshold = (double*)malloc(sizeof(double)*nr_class*(nr_class-1)/2);
				model_->nSV = 0;
				model_->cSV = (int*)Malloc(int, nr_class*(nr_class-1)/2);

				int p = 0;
				for(i=0;i<nr_class;i++){
					for(j=i+1;j<nr_class;j++){
						int ci = count[i], cj = count[j];
						r_size_d = (ci+cj)*param->m1_r;
						if((int)r_size_d == 0)
							model_->cSV[p] = ci+cj;
						else
							model_->cSV[p] = (int)r_size_d;
						model_->nSV += model_->cSV[p];
						p++;
					}
				}

				fprintf(stderr, "nSV = %d, cSV = ", model_->nSV);
				for(i = 0; i < p; i++)
					fprintf(stderr, "%d, ", model_->cSV[i]);
				fprintf(stderr, "\n");

				model_->SV = Malloc(feature_node *, model_->nSV);
				model_->w = Malloc(double, model_->nSV+(model_->nr_class*(model_->nr_class-1))/2);

				p = 0;
				int SV_begin = 0;
				int wp = 0;
				problem sub_prob;
				for(i=0;i<nr_class;i++)
					for(j=i+1;j<nr_class;j++){
						int si = start[i], sj = start[j];
						int ci = count[i], cj = count[j];
						int cim1 = (int)(ci*param->m1_r);
						int cjm1 = (int)(cj*param->m1_r);
						sub_prob.l = ci+cj;
						sub_prob.n = n;
						sub_prob.x = Malloc(feature_node *,sub_prob.l);
						sub_prob.y = Malloc(double,sub_prob.l);
						sub_prob.bias = prob->bias;

						double *w = (double*)Malloc(double, model_->cSV[p]+1);
						rp_size_d = model_->cSV[p]*param->m2_r;
						rp_size_i = (int)rp_size_d;
						if(cim1+cjm1 < model_->cSV[p])
							cim1 += model_->cSV[p]-cim1-cjm1;
						else if(cim1+cjm1 > model_->cSV[p])
							fprintf(stderr, "cimi&cjm1 are wrong.\n");

						int k;
						for(k=0;k<ci;k++)
						{
							sub_prob.x[k] = x[si+k];
							sub_prob.y[k] = +1;
						}
						for(k=0;k<cj;k++)
						{
							sub_prob.x[ci+k] = x[sj+k];
							sub_prob.y[ci+k] = -1;
						}

						if(param->init_sol != NULL)
							for(int k=0;k<model_->cSV[p];k++)
								w[k] = param->init_sol[p];
						else
							for(int k=0;k<=model_->cSV[p];k++)
								w[k] = 0;
					
						fprintf(stderr, "In submodel %d v.s. %d\n", i, j);
						if(!param->auto_threshold){
							solve_r_ls_svm_svc(&sub_prob, w, &model_->SV[SV_begin], param,
							weighted_C[i], weighted_C[j], model_->cSV[p], rp_size_i, cim1, cjm1, ci, cj, NULL);
						}else{
							solve_r_ls_svm_svc(&sub_prob, w, &model_->SV[SV_begin], param,
							weighted_C[i], weighted_C[j], model_->cSV[p], rp_size_i, cim1, cjm1, ci, cj, &model_->a_threshold[p]);
						}
						SV_begin += model_->cSV[p];

						for(int k=0;k<=model_->cSV[p];k++){
							model_->w[wp] = w[k];
							wp++;
						}

						free(sub_prob.x);
						free(sub_prob.y);
						free(w);
						p++;
					}
			}
			else if(param->solver_type == TRSVM){
				fprintf(stderr, "In TRSVM, class_nr = %d\n", nr_class);
				double r_size_d;
				if(param->auto_threshold)
					model_->a_threshold = (double*)malloc(sizeof(double)*nr_class*(nr_class-1)/2);
				model_->nSV = 0;
				model_->cSV = (int*)Malloc(int, nr_class*(nr_class-1)/2);

				int p = 0;
				for(i=0;i<nr_class;i++){
					for(j=i+1;j<nr_class;j++){
						int ci = count[i], cj = count[j];
						r_size_d = (ci+cj)*param->m1_r;
						if((int)r_size_d == 0)
							model_->cSV[p] = ci+cj;
						else
							model_->cSV[p] = (int)r_size_d;
						model_->nSV += model_->cSV[p];
						p++;
					}
				}

				fprintf(stderr, "nSV = %d, cSV = ", model_->nSV);
				for(i = 0; i < p; i++)
					fprintf(stderr, "%d, ", model_->cSV[i]);
				fprintf(stderr, "\n");

				model_->SV = Malloc(feature_node *, model_->nSV);
				model_->w = Malloc(double, model_->nSV+(model_->nr_class*(model_->nr_class-1))/2);

				p = 0;
				int SV_begin = 0;
				int wp = 0;
				problem sub_prob;
				for(i=0;i<nr_class;i++)
					for(j=i+1;j<nr_class;j++){
						int si = start[i], sj = start[j];
						int ci = count[i], cj = count[j];
						int cim1 = (int)(ci*param->m1_r);
						int cjm1 = (int)(cj*param->m1_r);
						sub_prob.l = ci+cj;
						sub_prob.n = n;
						sub_prob.x = Malloc(feature_node *,sub_prob.l);
						sub_prob.y = Malloc(double,sub_prob.l);
						sub_prob.bias = prob->bias;

						double *w = (double*)Malloc(double, model_->cSV[p]+1);
						if(cim1+cjm1 < model_->cSV[p])
							cim1 += model_->cSV[p]-cim1-cjm1;
						else if(cim1+cjm1 > model_->cSV[p])
							fprintf(stderr, "cimi&cjm1 are wrong.\n");

						int k;
						for(k=0;k<ci;k++)
						{
							sub_prob.x[k] = x[si+k];
							sub_prob.y[k] = +1;
						}
						for(k=0;k<cj;k++)
						{
							sub_prob.x[ci+k] = x[sj+k];
							sub_prob.y[ci+k] = -1;
						}

						if(param->init_sol != NULL)
							for(int k=0;k<model_->cSV[p];k++)
								w[k] = param->init_sol[p];
						else
							for(int k=0;k<=model_->cSV[p];k++)
								w[k] = 0;
					
						fprintf(stderr, "In submodel %d v.s. %d\n", i, j);
						if(!param->auto_threshold){
							solve_trsvm_svc(&sub_prob, w, &model_->SV[SV_begin], param,
							weighted_C[i], weighted_C[j], model_->cSV[p], cim1, cjm1, ci, cj, NULL);
						}else{
							solve_trsvm_svc(&sub_prob, w, &model_->SV[SV_begin], param,
							weighted_C[i], weighted_C[j], model_->cSV[p], cim1, cjm1, ci, cj, &model_->a_threshold[p]);
						}
						SV_begin += model_->cSV[p];

						for(int k=0;k<=model_->cSV[p];k++){
							model_->w[wp] = w[k];
							wp++;
						}

						free(sub_prob.x);
						free(sub_prob.y);
						free(w);
						p++;
					}
			}
			else if(nr_class == 2)
			{
				int k;
				problem sub_prob;
				sub_prob.l = l;
				sub_prob.n = n;
				sub_prob.x = Malloc(feature_node *,sub_prob.l);
				sub_prob.y = Malloc(double,sub_prob.l);

				for(k=0; k<sub_prob.l; k++)
					sub_prob.x[k] = x[k];

				model_->w=Malloc(double, w_size);

				int e0 = start[0]+count[0];
				k=0;
				for(; k<e0; k++)
					sub_prob.y[k] = +1;
				for(; k<sub_prob.l; k++)
					sub_prob.y[k] = -1;

				if(param->init_sol != NULL)
					for(i=0;i<w_size;i++)
						model_->w[i] = param->init_sol[i];
				else
					for(i=0;i<w_size;i++)
						model_->w[i] = 0;

				train_one(&sub_prob, param, model_->w, weighted_C[0], weighted_C[1]);
				free(sub_prob.x);
				free(sub_prob.y);
			}
			else
			{
				int k;
				problem sub_prob;
				sub_prob.l = l;
				sub_prob.n = n;
				sub_prob.x = Malloc(feature_node *,sub_prob.l);
				sub_prob.y = Malloc(double,sub_prob.l);

				for(k=0; k<sub_prob.l; k++)
					sub_prob.x[k] = x[k];
				model_->w=Malloc(double, w_size*nr_class);
				double *w=Malloc(double, w_size);
				for(i=0;i<nr_class;i++)
				{
					int si = start[i];
					int ei = si+count[i];

					k=0;
					for(; k<si; k++)
						sub_prob.y[k] = -1;
					for(; k<ei; k++)
						sub_prob.y[k] = +1;
					for(; k<sub_prob.l; k++)
						sub_prob.y[k] = -1;

					if(param->init_sol != NULL)
						for(j=0;j<w_size;j++)
							w[j] = param->init_sol[j*nr_class+i];
					else
						for(j=0;j<w_size;j++)
							w[j] = 0;

					train_one(&sub_prob, param, w, weighted_C[i], param->C);

					for(int j=0;j<w_size;j++)
						model_->w[j*nr_class+i] = w[j];
					free(sub_prob.x);
					free(sub_prob.y);
				}
				free(w);
			}

		}

		free(x);
		free(label);
		free(start);
		free(count);
		free(perm);
		free(weighted_C);
	}
	return model_;
}

void cross_validation(const problem *prob, const parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int,l);
	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct problem subprob;

		subprob.bias = prob->bias;
		subprob.n = prob->n;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct feature_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct model *submodel = train(&subprob,param);
		for(j=begin;j<end;j++)
			target[perm[j]] = predict(submodel,prob->x[perm[j]]);
		free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}

void find_parameter_C(const problem *prob, const parameter *param, int nr_fold, double start_C, double max_C, double *best_C, double *best_rate)
{
	// variables for CV
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int, l);
	double *target = Malloc(double, prob->l);
	struct problem *subprob = Malloc(problem,nr_fold);

	// variables for warm start
	double ratio = 2;
	double **prev_w = Malloc(double*, nr_fold);
	for(i = 0; i < nr_fold; i++)
		prev_w[i] = NULL;
	int num_unchanged_w = 0;
	struct parameter param1 = *param;
	void (*default_print_string) (const char *) = liblinear_print_string;

	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;

		subprob[i].bias = prob->bias;
		subprob[i].n = prob->n;
		subprob[i].l = l-(end-begin);
		subprob[i].x = Malloc(struct feature_node*,subprob[i].l);
		subprob[i].y = Malloc(double,subprob[i].l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob[i].x[k] = prob->x[perm[j]];
			subprob[i].y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob[i].x[k] = prob->x[perm[j]];
			subprob[i].y[k] = prob->y[perm[j]];
			++k;
		}

	}

	*best_rate = 0;
	if(start_C <= 0)
		start_C = calc_start_C(prob,param);
	param1.C = start_C;

	while(param1.C <= max_C)
	{
		//Output disabled for running CV at a particular C
		set_print_string_function(&print_null);

		for(i=0; i<nr_fold; i++)
		{
			int j;
			int begin = fold_start[i];
			int end = fold_start[i+1];

			param1.init_sol = prev_w[i];
			struct model *submodel = train(&subprob[i],&param1);

			int total_w_size;
			if(submodel->nr_class == 2)
				total_w_size = subprob[i].n;
			else
				total_w_size = subprob[i].n * submodel->nr_class;

			if(prev_w[i] == NULL)
			{
				prev_w[i] = Malloc(double, total_w_size);
				for(j=0; j<total_w_size; j++)
					prev_w[i][j] = submodel->w[j];
			}
			else if(num_unchanged_w >= 0)
			{
				double norm_w_diff = 0;
				for(j=0; j<total_w_size; j++)
				{
					norm_w_diff += (submodel->w[j] - prev_w[i][j])*(submodel->w[j] - prev_w[i][j]);
					prev_w[i][j] = submodel->w[j];
				}
				norm_w_diff = sqrt(norm_w_diff);

				if(norm_w_diff > 1e-15)
					num_unchanged_w = -1;
			}
			else
			{
				for(j=0; j<total_w_size; j++)
					prev_w[i][j] = submodel->w[j];
			}

			for(j=begin; j<end; j++)
				target[perm[j]] = predict(submodel,prob->x[perm[j]]);

			free_and_destroy_model(&submodel);
		}
		set_print_string_function(default_print_string);

		int total_correct = 0;
		for(i=0; i<prob->l; i++)
			if(target[i] == prob->y[i])
				++total_correct;
		double current_rate = (double)total_correct/prob->l;
		if(current_rate > *best_rate)
		{
			*best_C = param1.C;
			*best_rate = current_rate;
		}

		info("log2c=%7.2f\trate=%g\n",log(param1.C)/log(2.0),100.0*current_rate);
		num_unchanged_w++;
		if(num_unchanged_w == 3)
			break;
		param1.C = param1.C*ratio;
	}

	if(param1.C > max_C && max_C > start_C)
		info("warning: maximum C reached.\n");
	free(fold_start);
	free(perm);
	free(target);
	for(i=0; i<nr_fold; i++)
	{
		free(subprob[i].x);
		free(subprob[i].y);
		free(prev_w[i]);
	}
	free(prev_w);
	free(subprob);
}

double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int nr_w;
	if(nr_class==2 && model_->param.solver_type != MCSVM_CS) {
		nr_w = 1;
	} else if (model_->param.solver_type == R_LS_SVM) {
		nr_w = nr_class*(nr_class-1)/2;
	} else if (model_->param.solver_type == TRSVM) {
		nr_w = nr_class*(nr_class-1)/2;
	} else {
		nr_w = nr_class;
	}

	for(int i=0;i<nr_w;i++)
		dec_values[i] = 0;
	if(model_->param.solver_type == R_LS_SVM || model_->param.solver_type == TRSVM){
		int nSV = model_->nSV;
		double *Q_r = new double[nSV];
		feature_node **SV = model_->SV;

		Q_r = truncated_RBF(Q_r, &x, SV, 1, nSV, model_->param.threshold, model_->param.gamma, false, NULL);
		int w_p = 0;
		for(int i=0; i<nr_w; i++){
			int cSV_i = model_->cSV[i];
			for(int j=0; j<cSV_i; j++){
				if(model_->param.auto_threshold && Q_r[w_p] < model_->a_threshold[i]){
					w_p++;
					continue;
				}	
				dec_values[i] += w[w_p] * Q_r[w_p];
				w_p++;
			}
			if(dec_values[i] != 0)
				dec_values[i] += w[w_p] * model_->bias;
			w_p++;
		}
		int *vote = new int[nr_class];
		for(int i=0;i<nr_class;i++)
			vote[i] = 0;
		int p=0;
		for(int i=0;i<nr_class;i++){
			for(int j=i+1;j<nr_class;j++){
				if(dec_values[p] > 0)
					++vote[i];
				else if(dec_values[p] < 0)
					++vote[j];
				p++;
			}
		}
		int vote_max_idx = 0;
		for(int i=1;i<nr_class;i++)
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx = i;
		delete [] vote;
		delete [] Q_r;

		return model_->label[vote_max_idx];
	}else{
		const feature_node *lx=x;
		for(; (idx=lx->index)!=-1; lx++)
		{
			// the dimension of testing data may exceed that of training
			if(idx<=n)
				for(int i=0;i<nr_w;i++)
					dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
		}
		if(nr_class==2)
		{
			if(check_regression_model(model_))
				return dec_values[0];
			else
				return (dec_values[0]>0)?model_->label[0]:model_->label[1];
		}
		else
		{
			int dec_max_idx = 0;
			for(int i=1;i<nr_class;i++)
			{
				if(dec_values[i] > dec_values[dec_max_idx])
					dec_max_idx = i;
			}
			return model_->label[dec_max_idx];
		}
	}
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values;
	if (model_->param.solver_type == R_LS_SVM) {
		dec_values = Malloc(double, model_->nr_class*(model_->nr_class-1)/2);
	}
	else {
		dec_values = Malloc(double, model_->nr_class);
	}
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
{
	if(check_probability_model(model_))
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_w;
		if(nr_class==2)
			nr_w = 1;
		else
			nr_w = nr_class;

		double label=predict_values(model_, x, prob_estimates); //TODO handle the case of ono-against-one
		for(i=0;i<nr_w;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

		if(nr_class==2) // for binary classification
			prob_estimates[1]=1.-prob_estimates[0];
		else
		{
			double sum=0;
			for(i=0; i<nr_class; i++)
				sum+=prob_estimates[i];

			for(i=0; i<nr_class; i++)
				prob_estimates[i]=prob_estimates[i]/sum;
		}

		return label;
	}
	else
		return 0;
}

static const char *solver_type_table[]=
{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
	"L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL", "R_LS_SVM", "CSSVM", 
	"", "",
	"L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL", NULL
};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	int nr_w;
	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	if(param.solver_type == R_LS_SVM){
		w_size = model_->nSV+(model_->nr_class-1)*model_->nr_class/2;
		fprintf(fp, "gamma\n%.8g\n", param.gamma);
		if(param.auto_threshold){	
			fprintf(fp, "auto_threshold\n");
			for(i = 0; i < model_->nr_class*(model_->nr_class-1)/2; i++)
				fprintf(fp, "%.8g ", model_->a_threshold[i]);
			fprintf(fp, "\n");
		}
		else
			fprintf(fp, "threshold\n%.8g\n", param.threshold);
		fprintf(fp, "nSV\n%d\n", model_->nSV);
		fprintf(fp, "cSV");
		for(i = 0; i < model_->nr_class*(model_->nr_class-1)/2; i++)
			fprintf(fp, " %d", model_->cSV[i]);
		fprintf(fp, "\n");


		fprintf(fp, "SV\n");
		const feature_node * const *SV = model_->SV;
		for(int i = 0; i < model_->nSV; i++)
		{
			const feature_node *p = SV[i];

			int	index = 0;
			while(p != NULL && p[index].index != -1)
			{
				if(model_->bias >= 0 && p[index].index == n)
					break;
				fprintf(fp,"%d:%.8g ",p[index].index,p[index].value);
				index++;
			}
			fprintf(fp, "\n");
		}
	}
	if(param.solver_type == TRSVM){
		w_size = model_->nSV+(model_->nr_class-1)*model_->nr_class/2;
		fprintf(fp, "gamma\n%.8g\n", param.gamma);
		if(param.auto_threshold){	
			fprintf(fp, "auto_threshold\n");
			for(i = 0; i < model_->nr_class*(model_->nr_class-1)/2; i++)
				fprintf(fp, "%.8g ", model_->a_threshold[i]);
			fprintf(fp, "\n");
		}
		else
			fprintf(fp, "threshold\n%.8g\n", param.threshold);
		fprintf(fp, "nSV\n%d\n", model_->nSV);
		fprintf(fp, "cSV");
		for(i = 0; i < model_->nr_class*(model_->nr_class-1)/2; i++)
			fprintf(fp, " %d", model_->cSV[i]);
		fprintf(fp, "\n");


		fprintf(fp, "SV\n");
		const feature_node * const *SV = model_->SV;
		for(int i = 0; i < model_->nSV; i++)
		{
			const feature_node *p = SV[i];

			int	index = 0;
			while(p != NULL && p[index].index != -1)
			{
				if(model_->bias >= 0 && p[index].index == n)
					break;
				fprintf(fp,"%d:%.8g ",p[index].index,p[index].value);
				index++;
			}
			fprintf(fp, "\n");
		}
	}
	fprintf(fp, "w\n");
	if(param.solver_type == R_LS_SVM)
		for(int j = 0; j < model_->nSV+(model_->nr_class*(model_->nr_class-1)/2); j++)
			fprintf(fp, "%.16g ", model_->w[j]);
	else
		for(i=0; i<w_size; i++)
			for(int j=0; j<nr_w; j++)
				fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
	fprintf(fp, "\n");


	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var)do\
{\
	if (fscanf(_stream, _format, _var) != 1)\
	{\
		fprintf(stderr, "ERROR: fscanf failed to read the model\n");\
		EXIT_LOAD_MODEL()\
	}\
}while(0)
// EXIT_LOAD_MODEL should NOT end with a semicolon.
#define EXIT_LOAD_MODEL()\
{\
	setlocale(LC_ALL, old_locale);\
	free(model_->label);\
	free(model_);\
	free(old_locale);\
	return NULL;\
}
static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}
struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	bool R_LS_SVM_FLAG = false;
	bool TRSVM_FLAG = false;
	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	double gamma;
	int nSV;
	double threshold;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{
		FSCANF(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					if(i == 8)
						R_LS_SVM_FLAG = true;
					if(i == 9)
						TRSVM_FLAG = true;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				EXIT_LOAD_MODEL()
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			FSCANF(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			FSCANF(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"nSV")==0)
		{
			FSCANF(fp,"%d",&nSV);
			model_->nSV=nSV;
		}
		else if(strcmp(cmd,"cSV")==0)
		{
			int i = nr_class*(nr_class-1)/2;
			model_->cSV = (int*)malloc(i*sizeof(int));
			for(int j = 0; j < i; j++)
				FSCANF(fp,"%d",&model_->cSV[j]);
		}
		else if(strcmp(cmd,"bias")==0)
		{
			FSCANF(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			fprintf(stderr, "w\n");
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				FSCANF(fp,"%d",&model_->label[i]);
		}
		else if(strcmp(cmd,"gamma")==0){
			FSCANF(fp,"%lf",&gamma);
			param.gamma = gamma;
		}
		else if(strcmp(cmd,"auto_threshold")==0){
			param.threshold = 0;
			param.auto_threshold = true;
			model_->a_threshold = Malloc(double,nr_class*(nr_class-1)/2);
			for(i = 0; i < nr_class*(nr_class-1)/2; i++)
				FSCANF(fp, "%lf", &model_->a_threshold[i]);
		}
		else if(strcmp(cmd,"threshold")==0){
			FSCANF(fp,"%lf",&threshold);
			param.auto_threshold = false;
			param.threshold = threshold;
		}
		else if(strcmp(cmd,"SV")==0){
			int elements = 0;
			long pos = ftell(fp);

			max_line_len = 1024;
			line = Malloc(char,max_line_len);
			char *p,*endptr,*idx,*val;

			while(readline(fp)!=NULL)
			{
				p = strtok(line,":");
				while(1)
				{
					p = strtok(NULL,":");
					if(p == NULL)
						break;
					++elements;
				}
				++elements;
			}
			elements += model_->nSV;

			fseek(fp,pos,SEEK_SET);

			int l = model_->nSV;
			model_->SV = Malloc(feature_node*,l);
			feature_node *x_space = NULL;
			if(l>0) x_space = Malloc(feature_node,elements+model_->nSV);

			readline(fp);

			int j=0;
			bool first;
			for(i=0;i<l;i++)
			{
				readline(fp);
				model_->SV[i] = &x_space[j];
				first = true;

				while(1)
				{
					if(first){
						idx = strtok(line, ":");
						first = false;
					}
					else
						idx = strtok(NULL, ":");
					val = strtok(NULL, " \t");

					if(val == NULL)
						break;
					x_space[j].index = (int) strtol(idx,&endptr,10);
					x_space[j].value = strtod(val,&endptr);

					++j;
				}
				x_space[j].index = nr_feature+1;
				x_space[j++].value = model_->bias;
				x_space[j++].index = -1;
			}
			free(line);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			EXIT_LOAD_MODEL()
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2 && param.solver_type != MCSVM_CS)
		nr_w = 1;
	else if(R_LS_SVM_FLAG || TRSVM_FLAG)
		nr_w = nr_class*(nr_class-1)/2;
	else
		nr_w = nr_class;

	if(!R_LS_SVM_FLAG && !TRSVM_FLAG)
		model_->w = Malloc(double, w_size*nr_w);
	else
		model_->w = Malloc(double, nr_w+model_->nSV);

	if(R_LS_SVM_FLAG || TRSVM_FLAG)
		for(int k = 0; k < nr_w+model_->nSV; k++)
			FSCANF(fp, "%lf ", &model_->w[k]);
	else
		for(i=0; i<w_size; i++)
		{
			for(int j=0; j<nr_w; j++)
				FSCANF(fp, "%lf ", &model_->w[i*nr_w+j]);
			if (fscanf(fp, "\n") !=0)
			{
				fprintf(stderr, "ERROR: fscanf failed to read the model\n");
				EXIT_LOAD_MODEL()
			}
		}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

// use inline here for better performance (around 20% faster than the non-inline one)
static inline double get_w_value(const struct model *model_, int idx, int label_idx)
{
	int nr_class = model_->nr_class;
	int solver_type = model_->param.solver_type;
	const double *w = model_->w;

	if(idx < 0 || idx > model_->nr_feature)
		return 0;
	if(check_regression_model(model_))
		return w[idx];
	else
	{
		if(label_idx < 0 || label_idx >= nr_class)
			return 0;
		if(nr_class == 2 && solver_type != MCSVM_CS)
		{
			if(label_idx == 0)
				return w[idx];
			else
				return -w[idx];
		}
		else
			return w[idx*nr_class+label_idx];
	}
}

// feat_idx: starting from 1 to nr_feature
// label_idx: starting from 0 to nr_class-1 for classification models;
//            for regression models, label_idx is ignored.
double get_decfun_coef(const struct model *model_, int feat_idx, int label_idx)
{
	if(feat_idx > model_->nr_feature)
		return 0;
	return get_w_value(model_, feat_idx-1, label_idx);
}

double get_decfun_bias(const struct model *model_, int label_idx)
{
	int bias_idx = model_->nr_feature;
	double bias = model_->bias;
	if(bias <= 0)
		return 0;
	else
		return bias*get_w_value(model_, bias_idx, label_idx);
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
	if(param->init_sol != NULL)
		free(param->init_sol);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->p < 0)
		return "p < 0";

	if(param->solver_type != L2R_LR
		&& param->solver_type != L2R_L2LOSS_SVC_DUAL
		&& param->solver_type != L2R_L2LOSS_SVC
		&& param->solver_type != L2R_L1LOSS_SVC_DUAL
		&& param->solver_type != MCSVM_CS
		&& param->solver_type != L1R_L2LOSS_SVC
		&& param->solver_type != L1R_LR
		&& param->solver_type != L2R_LR_DUAL
		&& param->solver_type != L2R_L2LOSS_SVR
		&& param->solver_type != L2R_L2LOSS_SVR_DUAL
		&& param->solver_type != L2R_L1LOSS_SVR_DUAL
		&& param->solver_type != R_LS_SVM
		&& param->solver_type != TRSVM)
		return "unknown solver type";

	if(param->init_sol != NULL
		&& param->solver_type != L2R_LR && param->solver_type != L2R_L2LOSS_SVC)
		return "Initial-solution specification supported only for solver L2R_LR and L2R_L2LOSS_SVC";

	if(param->solver_type == R_LS_SVM){
		if(param->m1_r > 1 || param->m1_r <= 0)
			return "M must between (0, 1]";
		if(param->m2_r > 1 || param->m2_r <= 0)
			return "m must between (0, 1]";
	}

	return NULL;
}

int check_probability_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_LR ||
			model_->param.solver_type==L2R_LR_DUAL ||
			model_->param.solver_type==L1R_LR);
}

int check_regression_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_L2LOSS_SVR ||
			model_->param.solver_type==L2R_L1LOSS_SVR_DUAL ||
			model_->param.solver_type==L2R_L2LOSS_SVR_DUAL);
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}
