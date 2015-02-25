extern "C" {
#include "pendulum_QP_solver.h"
pendulum_QP_solver_FLOAT **f, **lb, **ub, **C, **e, **z;
}

#include <iostream>
#include <vector>
#include <ctime>

#include "../../../optcontrol/util/logging.h"

#include "pendulum_dynamics_by_hand.h" // Normally it's just pendulum_dynamics, but experimenting right now
using namespace pendulum;

#define INFTY 1e10

#define TIMESTEPS 15
const int T = TIMESTEPS;

#define X_DIM 2
#define U_DIM 1
#define VC_DIM 2
#define TOTAL_VARS

#include <eigen3/Eigen/Eigen>
using namespace Eigen;

#include "boost/preprocessor.hpp"

typedef Matrix<double, X_DIM, 1> VectorX;
typedef Matrix<double, U_DIM+VC_DIM, 1> VectorU;

typedef std::vector<VectorX> StdVectorX;
typedef std::vector<VectorU> StdVectorU;

namespace cfg {
const double improve_ratio_threshold = .1; // .1
const double min_approx_improve = 1e-4; // 1e-4
const double min_trust_box_size = 1e-4; // 1e-3
const double trust_shrink_ratio = .9; // .5
const double trust_expand_ratio = 1.1; // 1.5
const double cnt_tolerance = 1e-5; // 1e-5
const double penalty_coeff_increase_ratio = 10; // 10
const double initial_penalty_coeff = 1; // 1
const double initial_trust_box_size = .1; // 10
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

// fill in X in column major format with matrix XMat
inline void fill_col_major(double *X, const MatrixXd& XMat) {
	int idx = 0;
	int num_cols = XMat.cols();
	int num_rows = XMat.rows();
	for(size_t c = 0; c < num_cols; ++c) {
		for(size_t r = 0; r < num_rows; ++r) {
			X[idx++] = XMat(r, c);
		}
	}
}

void setup_state_vars(pendulum_QP_solver_params& problem, pendulum_QP_solver_output& output)
{

	/* Initialize problem inputs and outputs as double arrays */
	// problem inputs
	f = new pendulum_QP_solver_FLOAT*[T-1];
	lb = new pendulum_QP_solver_FLOAT*[T];
	ub = new pendulum_QP_solver_FLOAT*[T];
	C = new pendulum_QP_solver_FLOAT*[T-1];
	e = new pendulum_QP_solver_FLOAT*[T-1];

	// problem outputs
	z = new pendulum_QP_solver_FLOAT*[T];

	/* Link them via boost to something, IDK how this works */
#define SET_VARS(n) \
		lb[ BOOST_PP_SUB(n,1) ] = problem.lb##n ; \
		ub[ BOOST_PP_SUB(n,1) ] = problem.ub##n ; \
		z[ BOOST_PP_SUB(n,1) ] = output.z##n ;
#define BOOST_PP_LOCAL_MACRO(n) SET_VARS(n)
#define BOOST_PP_LOCAL_LIMITS (1, TIMESTEPS)
#include BOOST_PP_LOCAL_ITERATE()

#define SET_TMINUSONE_VARS(n) \
		f[ BOOST_PP_SUB(n,1) ] = problem.f##n ; \
		C[ BOOST_PP_SUB(n,1) ] = problem.C##n ; \
		e[ BOOST_PP_SUB(n,1) ] = problem.e##n ;
#define BOOST_PP_LOCAL_MACRO(n) SET_TMINUSONE_VARS(n)
#define BOOST_PP_LOCAL_LIMITS (1, TIMESTEPS-1)
#include BOOST_PP_LOCAL_ITERATE()

	// Initalize everything to infinity

	for(int t = 0; t < T-1; ++t) {
		fill_col_major(f[t], INFTY*Matrix<double,4*X_DIM+U_DIM+1,1>::Ones());
	}

	for(int t = 0; t < T-1; ++t) {
		fill_col_major(lb[t], INFTY*Matrix<double,4*X_DIM+U_DIM+1,1>::Ones());
		fill_col_major(ub[t], INFTY*Matrix<double,2*X_DIM+U_DIM+1,1>::Ones());
	}
	fill_col_major(lb[T-1], INFTY*Matrix<double,X_DIM+1,1>::Ones());
	fill_col_major(ub[T-1], INFTY*Matrix<double,X_DIM+1,1>::Ones());

	fill_col_major(C[0], INFTY*Matrix<double,2*X_DIM+1,4*X_DIM+U_DIM+1>::Ones());
	fill_col_major(e[0], INFTY*Matrix<double,2*X_DIM+1,1>::Ones());
	for(int t = 1; t < T-1; ++t) {
		fill_col_major(C[t], INFTY*Matrix<double,X_DIM+1,4*X_DIM+U_DIM+1>::Ones());
		fill_col_major(e[t], INFTY*Matrix<double,X_DIM+1,1>::Ones());
	}

	for(int t = 0; t < T-1; ++t) {
		fill_col_major(z[t], INFTY*Matrix<double, X_DIM+U_DIM+1, 1>::Ones());
	}
	fill_col_major(z[T-1], INFTY*Matrix<double, X_DIM+1, 1>::Ones());

}

void cleanup_state_vars() {
	delete[] f;
	delete[] lb;
	delete[] ub;
	delete[] C;
	delete[] e;
	delete[] z;
}

bool is_valid_inputs() {
	// Check if any of the values have not been touched, i.e. they are still infinity.

	for(int t = 0; t < T-1; ++t) {
		for(int i = 0; i < 4*X_DIM+U_DIM+1; ++i) {
			if (f[t][i] == INFTY) {return false;}
		}
	}

	for(int t = 0; t < T-1; ++t) {
		for(int i = 0; i < 4*X_DIM+U_DIM+1; ++i) {
			if (lb[t][i] == INFTY) {return false;}
		}
		for(int i = 0; i < 2*X_DIM+U_DIM+1; ++i) {
			if (ub[t][i] == INFTY) {return false;}
		}
	}
	for(int i = 0; i < X_DIM+1; ++i) {
		if (lb[T-1][i] == INFTY) {return false;}
	}
	for(int i = 0; i < X_DIM+1; ++i) {
		if (ub[T-1][i] == INFTY) {return false;}
	}


	for (int i = 0; i < (2*X_DIM+1)*(4*X_DIM+U_DIM+1); ++i) {
		if (C[0][i] == INFTY) {return false;}
	}
	for (int i = 0; i < 2*X_DIM+1; ++i) {
		if (e[0][i] == INFTY) {return false;}
	}
	for(int t = 1; t < T-1; ++t) {
		for (int i = 0; i < (X_DIM+1)*(4*X_DIM+U_DIM+1); ++i) {
			if (C[t][i] == INFTY) {return false;}
		}
		for (int i = 0; i < X_DIM+1; ++i) {
			if (e[t][i] == INFTY) {return false;}
		}
	}

	// Inputs are valid!
	return true;
}

// Fill in f using penalty coefficient
void fill_f(double penalty_coeff)
{
	VectorXd f_temp(4*X_DIM+U_DIM+1);
	for(int t = 0; t < T-1; ++t) {
		for(int i = 0; i < 4*X_DIM+U_DIM+1; ++i) {
			if (t==0 && i == X_DIM) { f_temp(i) = 1;}
			else if (i >= X_DIM+1+U_DIM+VC_DIM) { f_temp(i) = penalty_coeff; }
			else { f_temp(i) = 0;}
		}
		fill_col_major(f[t], f_temp);

		// std::cout << "f" << t << ":\n";
		// std::cout << f_temp << "\n";
	}
}

// Fill in lower bounds and upper bounds
void fill_lb_and_ub(StdVectorX& X, StdVectorU& U, double& delta, double trust_box_size, bounds_t bounds)
{
	VectorXd lb_temp(4*X_DIM+U_DIM+1);
	VectorXd ub_temp(2*X_DIM+U_DIM+1);

	for(int t = 0; t < T-1; ++t)
	{
		VectorX& xt = X[t];
		VectorU& ut = U[t];

		for(int i = 0; i < 4*X_DIM+U_DIM+1; ++i) {
			if (i < X_DIM) {
				//lb_temp(i) = MAX(bounds.x_min(i), xt[i] - trust_box_size);
				//ub_temp(i) = MIN(bounds.x_max(i), xt[i] + trust_box_size);
				lb_temp(i) = xt[i] - trust_box_size;
				ub_temp(i) = xt[i] + trust_box_size;
			}
			else if (i == X_DIM) {
				lb_temp(i) = MAX(bounds.delta_min, delta - trust_box_size);
				ub_temp(i) = MIN(bounds.delta_max, delta + trust_box_size);
			}
			else if (i > X_DIM && i < X_DIM+U_DIM+VC_DIM+1) {
				if (i < X_DIM + 1 + U_DIM) { // For normal controls
					lb_temp(i) = MAX(bounds.u_min(i-X_DIM-1), ut[i-X_DIM-1] - trust_box_size);
					ub_temp(i) = MIN(bounds.u_max(i-X_DIM-1), ut[i-X_DIM-1] + trust_box_size);
				} else { // For virtual controls
					//lb_temp(i) = bounds.virtual_control_min;
					//ub_temp(i) = bounds.virtual_control_max;
					lb_temp(i) = MAX(bounds.virtual_control_min, ut[i-X_DIM-1] - trust_box_size);
					ub_temp(i) = MIN(bounds.virtual_control_max, ut[i-X_DIM-1] + trust_box_size);
					//std::cout << lb_temp(i) << " " << ub_temp(i) << "\n";
				}
			}
			else { lb_temp(i) = 0; }
		}
		fill_col_major(lb[t], lb_temp);
		fill_col_major(ub[t], ub_temp);

		// std::cout << "lb" << t << ":\n";
		// std::cout << lb_temp << "\n";
		// std::cout << "ub" << t << ":\n";
		// std::cout << ub_temp << "\n";
	}

	//VectorX& xT = X[T-1];

	double eps = 1e-10;

	VectorXd lbT_temp(X_DIM+1);
	VectorXd ubT_temp(X_DIM+1);
	for(int i = 0; i < X_DIM; ++i) {
		lbT_temp(i) = bounds.x_goal(i) - eps;
		ubT_temp(i) = bounds.x_goal(i) + eps;
	}
	lbT_temp(X_DIM) = MAX(bounds.delta_min, delta - trust_box_size);
	ubT_temp(X_DIM) = MIN(bounds.delta_max, delta + trust_box_size);

	fill_col_major(lb[T-1], lbT_temp);
	fill_col_major(ub[T-1], ubT_temp);

	// std::cout << "lb" << T-1 << ":\n";
	// std::cout << lbT_temp << "\n";
	// std::cout << "ub" << T-1 << ":\n";
	// std::cout << ubT_temp << "\n";
}

// Fill in C and e by linearizing around point X, U, delta
void fill_in_C_and_e(StdVectorX& X, StdVectorU& U, double& delta, double trust_box_size, bounds_t bounds)
{
	VectorX& x0 = X[0];
	VectorU& u0 = U[0];

	VectorX xt1;
	Matrix<double, X_DIM, X_DIM+U_DIM+VC_DIM+1> jac = numerical_jacobian(continuous_dynamics, x0, u0, delta, dynamics_weights);
	Matrix<double, X_DIM, X_DIM> DH_X = jac.leftCols(X_DIM);
	Matrix<double, X_DIM, U_DIM+VC_DIM> DH_U = jac.middleCols(X_DIM, U_DIM+VC_DIM);
	Matrix<double, X_DIM, 1> DH_delta = jac.rightCols(1);

	Matrix<double, 2*X_DIM+1,4*X_DIM+U_DIM+1> C0_temp;
	Matrix<double, 2*X_DIM+1, 1> e0_temp;

	C0_temp.setZero();

	C0_temp.block<X_DIM,X_DIM>(0,0) = I;

	C0_temp.block<X_DIM,X_DIM>(X_DIM,0) = DH_X;
	C0_temp.block<X_DIM,1>(X_DIM,X_DIM) = DH_delta;
	C0_temp.block<X_DIM,U_DIM+VC_DIM>(X_DIM,X_DIM+1) = DH_U;
	C0_temp.block<X_DIM,X_DIM>(X_DIM,X_DIM+1+U_DIM+VC_DIM) = I;
	C0_temp.block<X_DIM,X_DIM>(X_DIM,X_DIM+1+U_DIM+VC_DIM+X_DIM) = minusI;
	C0_temp(2*X_DIM,X_DIM) = 1;

	fill_col_major(C[0], C0_temp);

	xt1 = rk45_DP(continuous_dynamics, x0, u0, delta, dynamics_weights);

	e0_temp.setZero();
	e0_temp.head(X_DIM) = bounds.x_start;
	e0_temp.block<X_DIM,1>(X_DIM,0) = -xt1 + DH_X*x0 + DH_U*u0 + delta*DH_delta;

	fill_col_major(e[0], e0_temp);

	Matrix<double, X_DIM+1,4*X_DIM+U_DIM+1> Ct_temp;
	Matrix<double, X_DIM+1, 1> et_temp;

	for(int t = 1; t < T-1; ++t)
	{
		VectorX& xt = X[t];
		VectorU& ut = U[t];

		xt1 = rk45_DP(continuous_dynamics, xt, ut, delta, dynamics_weights);
		jac = numerical_jacobian(continuous_dynamics, xt, ut, delta, dynamics_weights);
		DH_X = jac.leftCols(X_DIM);
		DH_U = jac.middleCols(X_DIM, U_DIM+VC_DIM);
		DH_delta = jac.rightCols(1);

		Ct_temp.setZero();

		Ct_temp.block<X_DIM,X_DIM>(0,0) = DH_X;
		Ct_temp.block<X_DIM,1>(0,X_DIM) = DH_delta;
		Ct_temp.block<X_DIM,U_DIM+VC_DIM>(0,X_DIM+1) = DH_U;
		Ct_temp.block<X_DIM,X_DIM>(0,X_DIM+1+U_DIM+VC_DIM) = I;
		Ct_temp.block<X_DIM,X_DIM>(0,X_DIM+1+U_DIM+VC_DIM+X_DIM) = minusI;
		Ct_temp(X_DIM,X_DIM) = 1;

		//std::cout << "t = " << t << "\n";
		//std::cout << "DH_X:\n" << DH_X << std::endl;
		//std::cout << "DH_U:\n" << DH_U << std::endl;
		//std::cout << "DH_delta:\n" << DH_delta << std::endl;

		//std::cout << Ct_temp.block<X_DIM+1,4*X_DIM+U_DIM+1>(0,0) << std::endl;
		//int num;
		//std::cin >> num;

		fill_col_major(C[t], Ct_temp);

		et_temp.setZero();
		et_temp.head(X_DIM) = -xt1 + DH_X*xt + DH_U*ut + delta*DH_delta;
		fill_col_major(e[t], et_temp);
	}

}

double computeObjective(double& delta, StdVectorX& X, StdVectorU& U) {
	return (delta);
}

double computeMerit(double& delta, StdVectorX& X, StdVectorU& U, double penalty_coeff) {
	double merit = computeObjective(delta, X, U);
	for(int t = 0; t < T-1; ++t) {
		VectorXd hval = dynamics_difference(continuous_dynamics, X[t], X[t+1], U[t], delta, dynamics_weights);
		merit += penalty_coeff*(hval.cwiseAbs()).sum();
	}
	return merit;
}


bool minimize_merit_function(StdVectorX& X, StdVectorU& U, double& delta, bounds_t bounds, double penalty_coeff,
		pendulum_QP_solver_params& problem, pendulum_QP_solver_output& output, pendulum_QP_solver_info& info) {

	// Initialize trust box size
	double trust_box_size;
	trust_box_size = cfg::initial_trust_box_size;

	// Initialize these things
	double merit = 0, optcost = 0;
	int index = 0;

	// Set best trajectory to input trajectory. This will allow us to linearize around the input trajectory for the first iteration.
	StdVectorX Xopt(T);
	StdVectorU Uopt(T-1);
	double deltaopt;

	bool success = true;

	LOG_INFO("delta: %f",delta);
	LOG_INFO("penalty coeff: %f",penalty_coeff);

	// fill in f. Constant across all iterations because the penalty is constant until we break from this "loop"
	fill_f(penalty_coeff);
	int sqp_iter;
	for(sqp_iter=0; sqp_iter < cfg::max_sqp_iterations; ++sqp_iter) {
		LOG_INFO("  sqp iter: %d",sqp_iter);

		merit = computeMerit(delta, X, U, penalty_coeff);

		// fill in C, e
		fill_in_C_and_e(X, U, delta, trust_box_size, bounds);

		for(int trust_iter=0; true; ++trust_iter) {
			LOG_INFO("       trust region size: %f",trust_box_size);

			// fill in lb, ub
			fill_lb_and_ub(X, U, delta, trust_box_size, bounds);

			if (!is_valid_inputs()) {
				LOG_FATAL("ERROR: invalid inputs");
				exit(0);
			}

			// call FORCES
			int exitflag = pendulum_QP_solver_solve(&problem, &output, &info);
			if (exitflag == 1) {
				optcost = info.pobj;
				deltaopt = z[0][X_DIM]; // Hard coded, I know the index of this
				//std::cout << z[0][0] << " " << z[0][1] << " " << z[0][2] << " " << z[0][3] << " " << z[0][4] << " " << z[0][5] << " " << z[0][6] << std::endl;
				//std::cout << z[1][0] << " " << z[1][1] << " " << z[1][2] << " " << z[1][3] << " " << z[1][4] << " " << z[1][5] << " " << z[1][6] << std::endl;
				for(int t=0; t < T; ++t) {
					index = 0;
					for(int i=0; i < X_DIM; ++i) {
						Xopt[t](i) = z[t][index++];
					}
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
				LOG_ERROR("Some problem in solver");
				success = false;
				return success;
			}

			double model_merit = optcost;
			double new_merit = computeMerit(deltaopt, Xopt, Uopt, penalty_coeff);
			double approx_merit_improve = merit - model_merit;
			double exact_merit_improve = merit - new_merit;
			double merit_improve_ratio = exact_merit_improve/approx_merit_improve;

			LOG_INFO("       approx improve: %.3f. exact improve: %.3f. ratio: %.3f", approx_merit_improve, exact_merit_improve, merit_improve_ratio);
			if (approx_merit_improve < -1) {
				LOG_INFO("Approximate merit function got worse (%.3e).", approx_merit_improve);
				LOG_INFO("Either convexification is wrong to zeroth order, or you're in numerical trouble");
				success = false;
				return success;
			} else if (approx_merit_improve < cfg::min_approx_improve) {
				LOG_INFO("Converged: y tolerance");
				X = Xopt; U = Uopt; delta = deltaopt;
				return success;
			} else if ((exact_merit_improve < 0) || (merit_improve_ratio < cfg::improve_ratio_threshold)) {
				trust_box_size = trust_box_size * cfg::trust_shrink_ratio;
			} else {
				trust_box_size = trust_box_size * cfg::trust_expand_ratio;
				X = Xopt; U = Uopt; delta = deltaopt;
				break;
			}

			if (trust_box_size < cfg::min_trust_box_size) {
				LOG_INFO("Converged x tolerance\n");
				return success;
			}

		} // end trust_region loop

	} // end second loop

	if (sqp_iter == cfg::max_sqp_iterations) {
		X = Xopt; U = Uopt; delta = deltaopt;
		LOG_INFO("Max number of SQP iterations reached")	
	}

	return success;

}

bool penalty_sqp(StdVectorX& X, StdVectorU& U, double& delta, bounds_t bounds,
		pendulum_QP_solver_params& problem, pendulum_QP_solver_output& output, pendulum_QP_solver_info& info) {
	double penalty_coeff = cfg::initial_penalty_coeff;
	int penalty_increases = 0;

	prev_delta = delta;

	bool success = true;

	while(penalty_increases < cfg::max_penalty_coeff_increases) {
		double constraint_violation = (computeMerit(delta, X, U, penalty_coeff) - computeObjective(delta, X, U))/penalty_coeff;		
		LOG_INFO("Initial constraint violation: %.5f", constraint_violation);

		success = minimize_merit_function(X, U, delta, bounds, penalty_coeff, problem, output, info);

		if (!success) {
			LOG_ERROR("Merit function not minimized successfully\n");
			break;
		}

		constraint_violation = (computeMerit(delta, X, U, penalty_coeff) - computeObjective(delta, X, U))/penalty_coeff;
		LOG_INFO("Constraint violation: %.5f\n", constraint_violation);

		if (constraint_violation <= cfg::cnt_tolerance) {
			break;
		} else if (constraint_violation > 1 && penalty_increases == 0) {

			if (delta > bounds.delta_max) {
				LOG_ERROR("Delta exceeds maximum allowed.\n");
				success = false;
				break;
			}

			delta = prev_delta + 0.1;
			//delta = MIN(delta, bounds.delta_max);
			prev_delta = delta;

			Matrix<double, X_DIM, T> init;
			for(int i = 0; i < X_DIM; ++i) {
				init.row(i).setLinSpaced(T, bounds.x_start(i), bounds.x_goal(i));
			}

			for(int t = 0; t < T; ++t) {
				X[t] = init.col(t);
			}

			// Initialize U variable
			//double c = (bounds.x_goal(3) - bounds.x_start(3)); // Trying this
			for(int t = 0; t < T-1; ++t) {
				U[t] = MatrixXd::Zero(U_DIM+VC_DIM, 1);
				//U[t] = c*MatrixXd::Ones(U_DIM, 1);
			}

		}
		else
		{
			penalty_increases++;
			penalty_coeff *= cfg::penalty_coeff_increase_ratio;
			if (penalty_increases == cfg::max_penalty_coeff_increases) {
				LOG_ERROR("Number of penalty coefficient increases exceeded maximum allowed.\n");
				success = false;
				break;
			}
		}

		
		// warm start?
		for(int t = 0; t < T-2; ++t) {
			X[t+1] = rk45_DP(continuous_dynamics, X[t], U[t], delta, dynamics_weights);
		}
	}

	return success;
}

int solve_BVP(double weights[], double pi[], double start_state[], double& delta, double virtual_control_max) {

	// Set pointer to weights
	dynamics_weights = weights;

	StdVectorX X(T);
	StdVectorU U(T-1);

	pendulum_QP_solver_params problem;
	pendulum_QP_solver_output output;
	pendulum_QP_solver_info info;
	setup_state_vars(problem, output);

	// Do bounds stuff here.
	// Note that we assume control_u_min = -1*control_u_max, and that it's only one number. This is what I found in Teodor's code
	bounds_t bounds;
	bounds.delta_min = 0;
	bounds.delta_max = 1000000;
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

	// Smart initialization
	//delta = std::min((bounds.x_start - bounds.x_goal).norm()/10, .5);
	delta = 0.001; // 0.01
	//std::cout << "Initial delta: " << delta << "\n";

	// Initialize X variable
	Matrix<double, X_DIM, T> init;
	for(int i = 0; i < X_DIM; ++i) {
		init.row(i).setLinSpaced(T, bounds.x_start(i), bounds.x_goal(i));
	}

	for(int t = 0; t < T; ++t) {
		X[t] = init.col(t);
	}

	// Initialize U variable
	//double c = (bounds.x_goal(3) - bounds.x_start(3)); // Trying this
	for(int t = 0; t < T-1; ++t) {
		U[t] = MatrixXd::Zero(U_DIM+VC_DIM, 1);
		//U[t] = c*MatrixXd::Ones(U_DIM, 1);
	}

	bool success = penalty_sqp(X, U, delta, bounds, problem, output, info);

	cleanup_state_vars();

	// Update pi with U[0] here. DO NOT ADD virtual controls. If failed, then add random controls.
	if (true) {
		for(int t = 0; t < T-1; ++t) {
			for(int i = 0; i < U_DIM; ++i) {
				pi[t*U_DIM + i] = U[t][i];
			}
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
