extern "C" {
#include "doublependulum_QP_solver_shooting.h"
doublependulum_QP_solver_shooting_FLOAT **f, **lb, **ub, **H, **z;
}

#include <iostream>
#include <vector>
#include <ctime>

#include "../../../optcontrol/util/logging.h"

#include "end_state.h"
#include "../doublependulum_dynamics_by_hand.h" // Normally it's just doublependulum_dynamics, but experimenting right now
using namespace doublependulum;

#define INFTY 1e10

#define TIMESTEPS 15
const int T = TIMESTEPS;

#define X_DIM 4
#define U_DIM 2
#define VC_DIM 4
#define TOTAL_VARS

#include <eigen3/Eigen/Eigen>
using namespace Eigen;

#include "boost/preprocessor.hpp"

typedef Matrix<double, X_DIM, 1> VectorX;
typedef Matrix<double, U_DIM+VC_DIM, 1> VectorU;
typedef Matrix<double, (U_DIM+VC_DIM+1)*(TIMESTEPS-1),1> VectorG;

typedef Matrix<double, (U_DIM+VC_DIM+1)*(TIMESTEPS-1), (U_DIM+VC_DIM+1)*(TIMESTEPS-1)> MatrixH;

typedef std::vector<VectorX> StdVectorX;
typedef std::vector<VectorU> StdVectorU;

namespace cfg {
const double improve_ratio_threshold = .1; // .1
const double min_approx_improve = 1e-4; // 1e-4
const double min_trust_box_size = 1e-4; // 1e-3
const double trust_shrink_ratio = .25; // .5
const double trust_expand_ratio = 1.25; // 1.5
const double cnt_tolerance = 1e-5; // 1e-5
const double penalty_coeff_increase_ratio = 10; // 10
const double initial_penalty_coeff = 1; // 1
const double initial_trust_box_size = 1; // 10
const int max_penalty_coeff_increases = 3; // 3
const int max_sqp_iterations = 100; // 100
}

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

Matrix<double, X_DIM,X_DIM> I = Matrix<double,X_DIM,X_DIM>::Identity(X_DIM,X_DIM);
Matrix<double, X_DIM,X_DIM> minusI = -I;

struct bounds_t {
	double delta_min, delta_max;
	double virtual_control_max;
	double virtual_control_min;
	//VectorX x_min, x_max;
	VectorU u_min, u_max;
	VectorX x_start;
	VectorX x_goal;
};

double *dynamics_weights;
double prev_delta;
double alpha;

// fill in X in column major format with matrix XMat
inline void fill_col_major(double *X, const MatrixXd& XMat) {
	int idx = 0;
	int num_cols = XMat.cols();
	int num_rows = XMat.rows();
	for(int c = 0; c < num_cols; ++c) {
		for(int r = 0; r < num_rows; ++r) {
			X[idx++] = XMat(r, c);
		}
	}
}

void setup_state_vars(doublependulum_QP_solver_shooting_params& problem, doublependulum_QP_solver_shooting_output& output)
{

	/* Initialize problem inputs and outputs as double arrays */
	// problem inputs
	f = new doublependulum_QP_solver_shooting_FLOAT*[T-1];
	lb = new doublependulum_QP_solver_shooting_FLOAT*[T-1];
	ub = new doublependulum_QP_solver_shooting_FLOAT*[T-1];
	H = new doublependulum_QP_solver_shooting_FLOAT*[T-1];

	// problem outputs
	z = new doublependulum_QP_solver_shooting_FLOAT*[T-1];

	/* Link them via boost to something, IDK how this works */
#define SET_TMINUSONE_VARS(n) \
		f[ BOOST_PP_SUB(n,1) ] = problem.f##n ; \
		H[ BOOST_PP_SUB(n,1) ] = problem.H##n ; \
		lb[ BOOST_PP_SUB(n,1) ] = problem.lb##n ; \
		ub[ BOOST_PP_SUB(n,1) ] = problem.ub##n ; \
		z[ BOOST_PP_SUB(n,1) ] = output.z##n ;
#define BOOST_PP_LOCAL_MACRO(n) SET_TMINUSONE_VARS(n)
#define BOOST_PP_LOCAL_LIMITS (1, TIMESTEPS-1)
#include BOOST_PP_LOCAL_ITERATE()

	// Initalize everything to infinity

	Matrix<double, X_DIM+U_DIM+1,1> INF_vec = INFTY*Matrix<double,X_DIM+U_DIM+1,1>::Ones();

	for(int t = 0; t < T-1; ++t) {

		fill_col_major(f[t], INF_vec);

		fill_col_major(H[t], INF_vec); // Diag

		fill_col_major(lb[t], INF_vec);
		fill_col_major(ub[t], INF_vec);

		fill_col_major(z[t], INF_vec);
	}

}

void cleanup_state_vars() {
	delete[] f;
	delete[] lb;
	delete[] ub;
	delete[] H;
	delete[] z;
}

bool is_valid_inputs() {
	// Check if any of the values have not been touched, i.e. they are still infinity.

	for(int t = 0; t < T-1; ++t) {
		for(int i = 0; i < X_DIM+U_DIM+1; ++i) {
			if (f[t][i] == INFTY) {return false;}
			if (H[t][i] == INFTY) {return false;}
			if (lb[t][i] == INFTY) {return false;}
			if (ub[t][i] == INFTY) {return false;}
		}
	}

	// Inputs are valid!
	return true;
}

// Quadratize cost function
double fill_f_and_H(StdVectorU& U, double delta, MatrixH& hess, VectorG& grad, bounds_t bounds, const double cost)
{

	VectorG diaghess = hess.diagonal();

	double constant_cost = 0, hessian_constant = 0, jac_constant = 0;
	
	// check if diag hessian
	for(int t = 0; t < T-1; ++t) {

		int index = t*(U_DIM+VC_DIM+1);

		// make sure it is psd
		for(int i = 0; i < (U_DIM+VC_DIM+1); ++i) {
			double val = diaghess(index+i);
			H[t][i] = max(0.0, val);
		}

		Matrix<double,(U_DIM+VC_DIM+1),1> zbar;
		zbar(0) = delta;
		zbar.block<U_DIM+VC_DIM,1>(1,0) = U[t];

		for (int i = 0; i < (U_DIM+VC_DIM+1); ++i) {
			hessian_constant += H[t][i]*zbar(i)*zbar(i);
			jac_constant -= grad(index+i)*zbar(i);

			f[t][i] = grad(index+i) - H[t][i]*zbar(i);
		}
		f[t][0] += alpha;
	}

	constant_cost = 0.5*hessian_constant + jac_constant + cost;
	return constant_cost;
}

// Fill in lower bounds and upper bounds
void fill_lb_and_ub(StdVectorU& U, double& delta, double trust_box_size, bounds_t bounds)
{
	VectorXd lb_temp(X_DIM+U_DIM+1);
	VectorXd ub_temp(X_DIM+U_DIM+1);

	for(int t = 0; t < T-1; ++t)
	{
		VectorU& ut = U[t];

		for(int i = 0; i < X_DIM+U_DIM+1; ++i) {

			// Delta
			if (i == 0) {
				lb_temp(i) = MAX(bounds.delta_min, delta - trust_box_size);
				ub_temp(i) = MIN(bounds.delta_max, delta + trust_box_size);				
			}

			// Controls/Torques
			else if (i < U_DIM+1) {
				lb_temp(i) = MAX(bounds.u_min(i-1), ut[i-1] - trust_box_size*control_max[i-1]); // scale of torques is crazy. Try accounting for them
				ub_temp(i) = MIN(bounds.u_max(i-1), ut[i-1] + trust_box_size*control_max[i-1]);

				//lb_temp(i) = MAX(bounds.u_min(i-1), ut[i-1] - trust_box_size);
				//ub_temp(i) = MIN(bounds.u_max(i-1), ut[i-1] + trust_box_size);				
			}

			// Virtual controls
			else {
				lb_temp(i) = MAX(bounds.virtual_control_min, ut[i-U_DIM-1] - trust_box_size);
				ub_temp(i) = MIN(bounds.virtual_control_max, ut[i-U_DIM-1] + trust_box_size);				
			}
		}

		fill_col_major(lb[t], lb_temp);
		fill_col_major(ub[t], ub_temp);

		// std::cout << "lb" << t << ":\n";
		// std::cout << lb_temp << "\n";
		// std::cout << "ub" << t << ":\n";
		// std::cout << ub_temp << "\n";
	}

}

void BFGS(StdVectorU& U, double delta, VectorG& grad, StdVectorU& Uopt, double deltaopt, VectorG& gradopt, MatrixH& hess) 
{
	VectorG s;
	s.setZero();

	int index = 0;
	for(int t=0; t < T-1; ++t) {
		s(index++) = deltaopt - delta;
		VectorU Udiff = Uopt[t] - U[t];
		for(int i=0; i < U_DIM+VC_DIM; ++i) { s(index++) = Udiff(i); }
	}
	
	VectorG y = gradopt - grad;
	double theta;
    VectorXd hess_s = hess*s;
	bool decision = s.dot(y) >= .2*s.dot(hess_s);
	
	if (decision) {
		theta = 1;
	} else {
		theta = (.8*s.dot(hess_s))/(s.dot(hess_s) - s.dot(y));
	}
    
    VectorXd r = theta*y + (1-theta)*hess_s;
	hess = hess - (hess_s*hess_s.transpose())/(s.dot(hess_s)) + (r*r.transpose())/s.dot(r);

	std::cout << hess.diagonal().transpose() << std::endl;
}

double computeMerit(double& delta, StdVectorU& U, bounds_t bounds) {

	double merit = alpha*delta + goal_cost(bounds.x_goal, bounds.x_start, U, delta, dynamics_weights);
	return merit;

}


bool minimize_merit_function(StdVectorU& U, double& delta, bounds_t bounds,
		doublependulum_QP_solver_shooting_params& problem, doublependulum_QP_solver_shooting_output& output, doublependulum_QP_solver_shooting_info& info) {

	// Initialize trust box size
	double trust_box_size;
	trust_box_size = cfg::initial_trust_box_size;

	// Initialize these things
	double cost = computeMerit(delta, U, bounds);
	double merit = cost, optcost = 0, cost_after_solve = 0;
	int index = 0;

	// Set best trajectory to input trajectory. This will allow us to linearize around the input trajectory for the first iteration.
	StdVectorU Uopt(T-1);
	double deltaopt;

	// Hessian stuff
	MatrixH hess = MatrixXd::Identity((U_DIM+VC_DIM+1)*(TIMESTEPS-1), (U_DIM+VC_DIM+1)*(TIMESTEPS-1));
	
	VectorG grad = goal_cost_numerical_gradient(bounds.x_goal, bounds.x_start, U, delta, dynamics_weights);
	VectorG gradopt = VectorG::Zero();

	bool success = true;

	LOG_INFO("delta: %f",delta);

	// fill in f. Constant across all iterations because the penalty is constant until we break from this "loop"
	int sqp_iter;
	for(sqp_iter=0; sqp_iter < cfg::max_sqp_iterations; ++sqp_iter) {

		LOG_INFO("  sqp iter: %d",sqp_iter);
		
		// fill in H, f
		double constant_cost = fill_f_and_H(U, delta, hess, grad, bounds, cost);
		
		
		for(int trust_iter=0; true; ++trust_iter) {
			LOG_INFO("       trust region size: %f",trust_box_size);

			// fill in lb, ub
			fill_lb_and_ub(U, delta, trust_box_size, bounds);

			if (!is_valid_inputs()) {
				LOG_FATAL("ERROR: invalid inputs");
				exit(0);
			}

			// call FORCES
			int exitflag = doublependulum_QP_solver_shooting_solve(&problem, &output, &info);
			if (exitflag == 1) {
				optcost = info.pobj;
				deltaopt = z[0][0]; // Hard coded, I know the index of this
				for(int t=0; t < T; ++t) {
					index = 0;
					index++; // Skip delta
					if (t < T-1) {
						for(int i=0; i < U_DIM; ++i) {
							Uopt[t](i) = z[t][index++];
						}
						for(int i=0; i < VC_DIM; ++i) {
							Uopt[t](i+U_DIM) = z[t][index++];
						}
					}
				}
				//int num;
				//std::cin >> num;
			} else {
				LOG_ERROR("Some problem in QP solver");
				success = false;
				return success;
			}

			double model_merit = optcost + constant_cost;
			double new_merit = computeMerit(deltaopt, Uopt, bounds);
			double approx_merit_improve = merit - model_merit;
			double exact_merit_improve = merit - new_merit;
			double merit_improve_ratio = exact_merit_improve/approx_merit_improve;

			LOG_INFO("		 merit: %.3f, model_merit: %.3f, optcost: %.3f, constant_cost: %.3f, new_merit: %.3f, delta_opt: %.3f", merit, model_merit, optcost, constant_cost, new_merit, deltaopt);

			LOG_INFO("       approx improve: %.3f. exact improve: %.3f. ratio: %.3f", approx_merit_improve, exact_merit_improve, merit_improve_ratio);
			if (approx_merit_improve < -1) {
				LOG_INFO("Approximate merit function got worse (%.3e).", approx_merit_improve);
				LOG_INFO("Either convexification is wrong to zeroth order, or you're in numerical trouble");
				success = false;
				return success;
			} else if (approx_merit_improve < cfg::min_approx_improve) {
				LOG_INFO("Converged: y tolerance");
				U = Uopt; delta = deltaopt;
				return success;
			} else if ((exact_merit_improve < 0) || (merit_improve_ratio < cfg::improve_ratio_threshold)) {
				trust_box_size = trust_box_size * cfg::trust_shrink_ratio;
			} else {
				trust_box_size = trust_box_size * cfg::trust_expand_ratio;
				
				cost_after_solve = computeMerit(deltaopt, Uopt, bounds);
				gradopt = goal_cost_numerical_gradient(bounds.x_goal, bounds.x_start, Uopt, deltaopt, dynamics_weights);
				// std::cout << "gradopt:\n" << gradopt << std::endl;

				// BFGS(U, delta, grad, Uopt, deltaopt, gradopt, hess);

				U = Uopt; delta = deltaopt;
				merit = cost_after_solve;
				cost = cost_after_solve;
				grad = gradopt;

				// LOG_INFO("		 cost_after_solve: %.3f", cost_after_solve);

				break; // for trust_region loop
			}

			if (trust_box_size < cfg::min_trust_box_size) {
				LOG_INFO("Converged x tolerance\n");
				return success;
			}

		} // end trust_region loop

	} // end second loop

	if (sqp_iter == cfg::max_sqp_iterations) {
		U = Uopt; delta = deltaopt;
		LOG_INFO("Max number of SQP iterations reached")	
	}

	return success;

}


bool penalty_sqp(StdVectorU& U, double& delta, bounds_t bounds,
		doublependulum_QP_solver_shooting_params& problem, doublependulum_QP_solver_shooting_output& output, doublependulum_QP_solver_shooting_info& info) {

	bool success = true;

	// while(penalty_increases < cfg::max_penalty_coeff_increases) {
	success = minimize_merit_function(U, delta, bounds, problem, output, info);

	if (!success) {
		LOG_ERROR("Merit function not minimized successfully\n");
	}

	LOG_INFO("Delta value: %.5f\n", delta);

	double cost = computeMerit(delta, U, bounds);
	LOG_INFO("	Final cost: %.3f", cost);

	return success;
}

int solve_BVP(double weights[], double pi[], double start_state[], double& delta, double virtual_control_max) {

	// Set pointer to weights
	dynamics_weights = weights;
	alpha = 0;

	StdVectorU U(T-1);

	doublependulum_QP_solver_shooting_params problem;
	doublependulum_QP_solver_shooting_output output;
	doublependulum_QP_solver_shooting_info info;
	setup_state_vars(problem, output);

	// Do bounds stuff here.
	// Note that we assume control_u_min = -1*control_u_max, and that it's only one number. This is what I found in Teodor's code
	bounds_t bounds;
	bounds.delta_min = 0.025;
	bounds.delta_max = 0.1;
	bounds.virtual_control_max = virtual_control_max;
	bounds.virtual_control_min = -1 * virtual_control_max;
	for(int i = 0; i < U_DIM; ++i) {
		bounds.u_min(i) = control_min[i]; // control_min and control_max is found in the dynamics file
		bounds.u_max(i) = control_max[i];
	}
	for(int i = 0; i < X_DIM; ++i) {
		bounds.x_start(i) = start_state[i];
		bounds.x_goal(i) = target_state[i]; // target_state is found in the dynamics file
	}

	// Initialization
	delta = 0.05; // 0.01

	// Initialize U variable
	//double c = (bounds.x_goal(3) - bounds.x_start(3)); // Trying this
	for(int t = 0; t < T-1; ++t) {
		// U[t] = MatrixXd::Zero(U_DIM+VC_DIM, 1);
		U[t] = uniform(-.1,.1)*MatrixXd::Ones(U_DIM+VC_DIM, 1);
	}

	bool success = penalty_sqp(U, delta, bounds, problem, output, info);

	cleanup_state_vars();

	// Update pi with U[0] here. DO NOT ADD virtual controls
	for(int t = 0; t < T-1; ++t) {
		for(int i = 0; i < U_DIM; ++i) {
			pi[t*U_DIM + i] = U[t][i];
		}
	}

	if (success) {
		// std::cout << "Final state:\n" << X[T-1] << "\n";
		//printf("Delta: %.5f\n", delta);
		return 1;
	} else {
		return 0;
	}

}

/*
 *  This function serves as a testing function for the methods written above
 */
/*
int main() {

	//double weights[NW] = { 0.0, -1.625, 0.0, 0.0, 0.375, 0.0, 0.0, -58.92, 0.0, 0.0, -0.375, -60.0, 0.6, 0.0, 0.0, -3.25, 0.0, 0.0, 0.0, 0.0, 0.75, -0.4, 0.0, 40.0, 7.365, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	double weights[NW] = {  3.03802826e+00,   1.18043322e+00,   8.10109424e-01,
        -4.88073929e-01,  -1.18013893e+00,  -3.02297203e+00,
        -2.42441014e+01,   9.47570297e+01,  -7.37037274e+00,
        -4.68601086e+01,  -3.55909025e+00,   7.33274499e+00,
         2.46211743e+01,   6.30525928e+00,  -8.41626412e-01,
         9.54373508e+00,   3.42934357e+00,  -6.56533059e-01,
        -1.12297446e-01,  -3.48279828e+00,  -9.72986865e+00,
        -2.05647992e+00,   3.19664052e+02,  -4.19469819e+01,
        -1.59554250e+02,   3.74443658e+01,   4.23969512e+01,
         1.90501274e+00,  -7.55190324e+01,   5.95848247e-01,
        -6.51835405e+00,  -1.85869964e+00,  -6.43799794e-01,
        -3.22698788e+00,   2.01335911e+00,   6.81133641e+00,
        -1.02427992e+02,  -2.91356940e+02,   4.39305980e+01,
         1.50203603e+02,  -3.51942923e+01,  -4.36170239e+01,
         1.05957744e+02,   6.75646574e+01,   6.30519926e-01,
        -3.40211031e+01,  -5.74340547e+00,   2.49740357e-01,
         2.83997849e+00,   6.30068109e+00,   3.63443434e+01,
        -8.03598497e+00,  -6.26963339e+02,   3.47655480e+02,
         3.23864826e+02,  -3.79694918e+01,  -3.50992409e+02,
         3.70999666e+00,   6.84871118e+01,  -7.12666479e-01};

	// Start and goal state
	//VectorX x_start, x_goal;

	//double initial_state[NX] = { 0.0, 0.0, -0.00116139991132, 0.312921810594 };
	//double initial_state[NX] = { 0.0, 0.0, 0.010977943328, 0.120180320014 };

	//x_start << 0.0, 0.0, -0.00116139991132, 0.312921810594;
	//x_start << 0.0, 0.0, 3.0, 0.0;
	//x_goal << 0.0, 0.0, 3.14159265359, 0.0;


	//double start_state[X_DIM] = {0.0, 0.0, -0.00116139991132, 0.312921810594};
	//double start_state[X_DIM] = {0.0, 0.0, 3.0, 0.0};
	//double start_state[X_DIM] = {0.0, 0.0, 0.0, -100.0};
	double start_state[X_DIM] = {-1.3017649 ,  0.4954675 , -0.01133658, -0.15953714};
	//double start_state[X_DIM] = {0.0, 0.0, 0.0, 0.312921810594};
	double pi[U_DIM];
	double delta = 1;

	double model_slack_bounds = 1;

	// Time it
	std::clock_t start;
	start = std::clock();

	bool success = solve_BVP(weights, pi, start_state, delta, model_slack_bounds);

	double solvetime = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

	printf("Solve time: %5.3f s\n", solvetime);

	if (success) {
		printf("Success!!\n");
		VectorXd PI(U_DIM);
		for (int i = 0; i < U_DIM; i++) {
			PI(i) = pi[i];
		}
		printf("pi:\n");
		std::cout << PI << "\n";
	} else {
		printf("Failure...\n");
		VectorXd PI(U_DIM);
		for (int i = 0; i < U_DIM; i++) {
			PI(i) = pi[i];
		}
		printf("pi:\n");
		std::cout << PI << "\n";
	}

	cleanup_state_vars();
}
*/
