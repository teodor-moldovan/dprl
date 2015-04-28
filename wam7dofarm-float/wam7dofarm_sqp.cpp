extern "C" {
#include "wam7dofarm_QP_solver.h"
wam7dofarm_QP_solver_FLOAT **f, **lb, **ub, **C, **e, **z;
wam7dofarm_QP_solver_FLOAT *A_T, *b_T;
}

#include <iostream>
#include <vector>
#include <ctime>

#include "../../optcontrol/util/logging.h"

#include "wam7dofarm_dynamics_by_hand.h" // Normally it's just wam7dofarm_dynamics, but experimenting right now
using namespace wam7dofarm;

#define INFTY 1000000

#define TIMESTEPS 8
const int T = TIMESTEPS;

#define X_DIM 14
#define U_DIM 7
#define VC_DIM 14
#define TOTAL_VARS

#include <eigen3/Eigen/Eigen>
using namespace Eigen;

#include "boost/preprocessor.hpp"

typedef Matrix<float, X_DIM, 1> VectorX;
typedef Matrix<float, U_DIM+VC_DIM, 1> VectorU;

typedef std::vector<VectorX> StdVectorX;
typedef std::vector<VectorU> StdVectorU;

namespace cfg {
const float improve_ratio_threshold = 0.05; // .1
const float min_approx_improve = 1e-4; // 1e-4
const float min_trust_box_size = 1e-3; // 1e-3
const float trust_shrink_ratio = .9; // .4
const float trust_expand_ratio = 1.1; // 1.3
const float cnt_tolerance = 1e-3; // 1e-5
const float penalty_coeff_increase_ratio = 10; // 10
const float initial_penalty_coeff = 1; // 1
const float initial_trust_box_size = 1; // 10
const int max_penalty_coeff_increases = 1; // 3
const int max_sqp_iterations = 100; // 150
}

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


struct bounds_t {
	float delta_min, delta_max;
	float virtual_control_max;
	float virtual_control_min;
	VectorX x_min, x_max;
	VectorU u_min, u_max;
	VectorX x_start;
	VectorX x_goal;
	Vector3f pos_goal;
	Vector3f vel_goal;
};

float *dynamics_weights;
float *prev_pi;
float prev_delta;
float QP_count = 0;
float goal_radius = 0.05;
float vc_max = 0;

// fill in X in column major format with matrix XMat
inline void fill_col_major(float *X, const MatrixXf& XMat) {
	int idx = 0;
	int num_cols = (int)XMat.cols();
	int num_rows = (int)XMat.rows();
	for(int c = 0; c < num_cols; ++c) {
		for(int r = 0; r < num_rows; ++r) {
			X[idx++] = XMat(r, c);
		}
	}
}

void setup_state_vars(wam7dofarm_QP_solver_params& problem, wam7dofarm_QP_solver_output& output)
{

	/* Initialize problem inputs and outputs as float arrays */
	// problem inputs
	f = new wam7dofarm_QP_solver_FLOAT*[T];
	lb = new wam7dofarm_QP_solver_FLOAT*[T];
	ub = new wam7dofarm_QP_solver_FLOAT*[T];
	C = new wam7dofarm_QP_solver_FLOAT*[T-1];
	e = new wam7dofarm_QP_solver_FLOAT*[T-1];
	// A_T = new wam7dofarm_QP_solver_FLOAT*; // Simply last time step
	// b_T = new wam7dofarm_QP_solver_FLOAT*;

	// problem outputs
	z = new wam7dofarm_QP_solver_FLOAT*[T];

	/* Link them via boost to something, IDK how this works */
#define SET_VARS(n) \
		f[ BOOST_PP_SUB(n,1) ] = problem.f##n ; \
		lb[ BOOST_PP_SUB(n,1) ] = problem.lb##n ; \
		ub[ BOOST_PP_SUB(n,1) ] = problem.ub##n ; \
		z[ BOOST_PP_SUB(n,1) ] = output.z##n ;
#define BOOST_PP_LOCAL_MACRO(n) SET_VARS(n)
#define BOOST_PP_LOCAL_LIMITS (1, TIMESTEPS)
#include BOOST_PP_LOCAL_ITERATE()

#define SET_TMINUSONE_VARS(n) \
		C[ BOOST_PP_SUB(n,1) ] = problem.C##n ; \
		e[ BOOST_PP_SUB(n,1) ] = problem.e##n ;
#define BOOST_PP_LOCAL_MACRO(n) SET_TMINUSONE_VARS(n)
#define BOOST_PP_LOCAL_LIMITS (1, TIMESTEPS-1)
#include BOOST_PP_LOCAL_ITERATE()

#define SET_OTHER_VARS(n) \
		A_T = problem.A##n ; \
		b_T = problem.b##n ;
#define BOOST_PP_LOCAL_MACRO(n) SET_OTHER_VARS(n)
#define BOOST_PP_LOCAL_LIMITS (TIMESTEPS, TIMESTEPS)
#include BOOST_PP_LOCAL_ITERATE()


	// A_T = problem.A10;
	// b_T = problem.b10;


	// Initalize everything to infinity

	for(int t = 0; t < T-1; ++t) {
		fill_col_major(f[t], INFTY*Matrix<float,4*X_DIM+U_DIM+1,1>::Ones());
	}
	fill_col_major(f[T-1], INFTY*Matrix<float,X_DIM+1+6+6,1>::Ones());

	for(int t = 0; t < T-1; ++t) {
		fill_col_major(lb[t], INFTY*Matrix<float,4*X_DIM+U_DIM+1,1>::Ones());
		fill_col_major(ub[t], INFTY*Matrix<float,2*X_DIM+U_DIM+1,1>::Ones());
	}
	fill_col_major(lb[T-1], INFTY*Matrix<float,X_DIM+1+6+6,1>::Ones());
	fill_col_major(ub[T-1], INFTY*Matrix<float,X_DIM+1,1>::Ones());

	fill_col_major(C[0], INFTY*Matrix<float,2*X_DIM+1,4*X_DIM+U_DIM+1>::Ones());
	fill_col_major(e[0], INFTY*Matrix<float,2*X_DIM+1,1>::Ones());
	for(int t = 1; t < T-1; ++t) {
		fill_col_major(C[t], INFTY*Matrix<float,X_DIM+1,4*X_DIM+U_DIM+1>::Ones());
		fill_col_major(e[t], INFTY*Matrix<float,X_DIM+1,1>::Ones());
	}

	fill_col_major(A_T, INFTY*Matrix<float,12,X_DIM+1+6+6>::Ones());
	fill_col_major(b_T, INFTY*Matrix<float,12,1>::Ones());

	for(int t = 0; t < T-1; ++t) {
		fill_col_major(z[t], INFTY*Matrix<float, X_DIM+U_DIM+1, 1>::Ones());
	}
	fill_col_major(z[T-1], INFTY*Matrix<float, X_DIM+1, 1>::Ones());

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
	for(int i = 0; i < X_DIM+1+6+6; ++i) {
		if (f[T-1][i] == INFTY) {return false;}
	}

	for(int t = 0; t < T-1; ++t) {
		for(int i = 0; i < 4*X_DIM+U_DIM+1; ++i) {
			if (lb[t][i] == INFTY) {return false;}
		}
		for(int i = 0; i < 2*X_DIM+U_DIM+1; ++i) {
			if (ub[t][i] == INFTY) {return false;}
		}
	}
	for(int i = 0; i < X_DIM+1+6+6; ++i) {
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

	for(int i = 0; i < 12*(X_DIM+1+6+6); ++i) {
		if (A_T[i] == INFTY) {return false;}
	}

	for(int i = 0; i < 12; ++i) {
		if (b_T[i] == INFTY) {return false;}
	}


	// Inputs are valid!
	return true;
}

// Fill in f using penalty coefficient
void fill_f(float penalty_coeff)
{
	VectorXf f_temp(4*X_DIM+U_DIM+1);
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

	VectorXf fT_temp(X_DIM+1+6+6);
	for(int i = 0; i < X_DIM+1+6+6; ++i) {
		if (i >= X_DIM+1) { fT_temp(i) = penalty_coeff; }
		else { fT_temp(i) = 0;}
	}
	// std::cout << "f_T" << ":\n";
	// std::cout << fT_temp << "\n";
	fill_col_major(f[T-1], fT_temp);

}

// Fill in lower bounds and upper bounds
void fill_lb_and_ub(StdVectorX& X, StdVectorU& U, float& delta, float trust_box_size, bounds_t bounds)
{
	VectorXf lb_temp(4*X_DIM+U_DIM+1);
	VectorXf ub_temp(2*X_DIM+U_DIM+1);

	for(int t = 0; t < T-1; ++t)
	{
		VectorX& xt = X[t];
		VectorU& ut = U[t];

		for(int i = 0; i < 4*X_DIM+U_DIM+1; ++i) { // Don't bound velocities
			if (i < X_DIM/2) {
				lb_temp(i) = xt[i] - trust_box_size;
				ub_temp(i) = xt[i] + trust_box_size;
			}
			else if (i >= X_DIM/2 && i < X_DIM) { 
				lb_temp(i) = MAX(bounds.x_min(i), xt[i] - trust_box_size);
				ub_temp(i) = MIN(bounds.x_max(i), xt[i] + trust_box_size);
			}
			else if (i == X_DIM) {
				lb_temp(i) = MAX(bounds.delta_min, delta - trust_box_size);
				ub_temp(i) = MIN(bounds.delta_max, delta + trust_box_size);
			}
			else if (i > X_DIM && i < X_DIM+U_DIM+VC_DIM+1) {
				if (i < X_DIM + 1 + U_DIM) { // For normal controls
					lb_temp(i) = MAX(bounds.u_min(i-X_DIM-1), ut[i-X_DIM-1] - trust_box_size*control_max[i-X_DIM-1]); // scale of torques is crazy. Try accounting for them
					ub_temp(i) = MIN(bounds.u_max(i-X_DIM-1), ut[i-X_DIM-1] + trust_box_size*control_max[i-X_DIM-1]);
				} else { // For virtual controls
					//lb_temp(i) = bounds.virtual_control_min;
					//ub_temp(i) = bounds.virtual_control_max;
					lb_temp(i) = MAX(bounds.virtual_control_min, ut[i-X_DIM-1] - vc_max*trust_box_size);
					ub_temp(i) = MIN(bounds.virtual_control_max, ut[i-X_DIM-1] + vc_max*trust_box_size);
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

	VectorX& xT = X[T-1];

	// float eps = 1e-10;

	VectorXf lbT_temp(X_DIM+1+6+6); lbT_temp.setZero();
	VectorXf ubT_temp(X_DIM+1);
	// for(int i = 0; i < X_DIM; ++i) {
	// 	lbT_temp(i) = bounds.x_goal(i) - eps;
	// 	ubT_temp(i) = bounds.x_goal(i) + eps;
	// }
	for(int i = 0; i < X_DIM; ++i) { // Don't bound velocities
		if (i < X_DIM/2) {
			lbT_temp(i) = xT[i] - trust_box_size;
			ubT_temp(i) = xT[i] + trust_box_size;
		}
		else if (i >= X_DIM/2 && i < X_DIM) { 
			lbT_temp(i) = MAX(bounds.x_min(i), xT[i] - trust_box_size);
			ubT_temp(i) = MIN(bounds.x_max(i), xT[i] + trust_box_size);
		}
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
void fill_in_C_and_e(StdVectorX& X, StdVectorU& U, float& delta, float trust_box_size, bounds_t bounds)
{
	VectorX& x0 = X[0];
	VectorU& u0 = U[0];

	VectorX xt1;
	Matrix<float, X_DIM, X_DIM+U_DIM+VC_DIM+1> jac = numerical_jacobian(continuous_dynamics, x0, u0, delta, dynamics_weights);
	Matrix<float, X_DIM, X_DIM> DH_X = jac.leftCols(X_DIM);
	Matrix<float, X_DIM, U_DIM+VC_DIM> DH_U = jac.middleCols(X_DIM, U_DIM+VC_DIM);
	Matrix<float, X_DIM, 1> DH_delta = jac.rightCols(1);

	Matrix<float, 2*X_DIM+1,4*X_DIM+U_DIM+1> C0_temp;
	Matrix<float, 2*X_DIM+1, 1> e0_temp;
	
	Matrix<float, X_DIM,X_DIM> I = Matrix<float,X_DIM,X_DIM>::Identity(X_DIM,X_DIM);
	Matrix<float, X_DIM,X_DIM> minusI = -I;

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

	Matrix<float, X_DIM+1,4*X_DIM+U_DIM+1> Ct_temp;
	Matrix<float, X_DIM+1, 1> et_temp;

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

void fill_A_T_and_b_T(StdVectorX& X, StdVectorU& U, float& delta, float trust_box_size, bounds_t bounds) {

	VectorX& x_T = X[T-1];
	Vector3f x_T_pos_eval;
	Vector3f x_T_vel_eval;

	// Evaluate the actual end effector position and velocity
	x_T_pos_eval = end_effector_pos(x_T);
	x_T_vel_eval = end_effector_lin_vel(x_T);
	
	// Cache 6x6 identity matrices
	Matrix<float, 6,6> I = Matrix<float,6,6>::Identity(6,6);
	Matrix<float, 6,6> minusI = -I;

	// Create matrices that I will need, start by setting them to zero
	Matrix<float, 12,X_DIM+1+6+6> A_T_temp;
	Matrix<float, 12, 1> b_T_temp;
	
	Matrix<float, 3, 1> buffer = Matrix<float, 3, 1>::Ones()*goal_radius;	

	A_T_temp.setZero(); b_T_temp.setZero();

	// Numerically calculate the jacobian of the end effector functions of the last joint angle
	Matrix<float, 3, X_DIM> pos_jac = general_numerical_jacobian(end_effector_pos, x_T, 3);
	Matrix<float, 3, X_DIM> vel_jac = general_numerical_jacobian(end_effector_lin_vel, x_T, 3);

	//std::cout << "EE pos jac:\n" << pos_jac << "\n";
	//std::cout << "EE lin vel jac:\n" << vel_jac << "\n";
	
	// Update the top half of A_T, the bottom half is just -1 * (top_half)
	A_T_temp.block<3, X_DIM>(0,0) = pos_jac;
	A_T_temp.block<3, X_DIM>(3,0) = vel_jac;
	A_T_temp.block<3, X_DIM>(6,0) = -pos_jac;
	A_T_temp.block<3, X_DIM>(9,0) = -vel_jac;

    A_T_temp.block<6,6>(0, X_DIM+1) = minusI;        
    A_T_temp.block<6,6>(6, X_DIM+1+6) = minusI;
    // A_T_temp.block<3,3>(0, X_DIM+1) = -1 * Matrix<float, 3, 3>::Identity(3,3);        
    // A_T_temp.block<3,3>(6, X_DIM+1+6) = -1 * Matrix<float, 3, 3>::Identity(3,3);
	
	b_T_temp.block<3,1>(0,0) = bounds.pos_goal - x_T_pos_eval + pos_jac*x_T + buffer;
	b_T_temp.block<3,1>(3,0) = bounds.vel_goal - x_T_vel_eval + vel_jac*x_T + buffer;

	b_T_temp.block<3,1>(6,0) = -bounds.pos_goal + x_T_pos_eval - pos_jac*x_T + buffer;
	b_T_temp.block<3,1>(9,0) = -bounds.vel_goal + x_T_vel_eval - vel_jac*x_T + buffer;

	// std::cout << "A_T_temp:\n" << A_T_temp << "\n";
	// std::cout << "b_T_temp:\n" << b_T_temp << "\n";

	// Fill them to the problem definition for FORCES
	
	// Sanity check to see if QP solve works if A_T and b_T are set to zero
	//A_T_temp.setZero();
	//b_T_temp.setZero();

	fill_col_major(A_T, A_T_temp);
	fill_col_major(b_T, b_T_temp);

}

float computeObjective(float& delta, StdVectorX& X, StdVectorU& U) {
	return (delta);
}

float computeMerit(float& delta, StdVectorX& X, StdVectorU& U, float penalty_coeff, bounds_t bounds) {

	float merit = computeObjective(delta, X, U);
	VectorXf hval;
	for(int t = 0; t < T-1; ++t) { // Same as just adding slacks in each state. Idk why I didn't do that but oh well.
		// std::cout << "X:\n" << X[t].transpose() << "\n";
		// std::cout << "U:\n" << U[t].transpose() << "\n";		
		hval = dynamics_difference(continuous_dynamics, X[t], X[t+1], U[t], delta, dynamics_weights);
		merit += penalty_coeff*(hval.cwiseAbs()).sum();
	}

	// Last time step slacks
	Vector3f buffer;
	buffer << goal_radius, goal_radius, goal_radius;

	VectorXf pos_viol = bounds.pos_goal - end_effector_pos(X[T-1]);
	VectorXf vel_viol = bounds.vel_goal - end_effector_lin_vel(X[T-1]);

	pos_viol = pos_viol.cwiseAbs() - buffer;
	vel_viol = vel_viol.cwiseAbs() - buffer;

	for(int i = 0; i < 3; ++i) {
		if (pos_viol(i) < 0) {
			pos_viol(i) = 0;
		}		
		if (vel_viol(i) < 0) {
			vel_viol(i) = 0;
		}

		// pos_viol(i) = std::max(0, pos_viol(i));	
		// vel_viol(i) = std::max(0, vel_viol(i));	
	}
	

	merit += penalty_coeff*(pos_viol).sum();
	merit += penalty_coeff*(vel_viol).sum();

	return merit;
}


bool minimize_merit_function(StdVectorX& X, StdVectorU& U, float& delta, bounds_t bounds, float penalty_coeff,
		wam7dofarm_QP_solver_params& problem, wam7dofarm_QP_solver_output& output, wam7dofarm_QP_solver_info& info) {

	// Initialize trust box size
	float trust_box_size;
	trust_box_size = cfg::initial_trust_box_size;
	
	//cfg::initial_trust_box_size = 0.1;


	// Initialize these things
	float merit = 0, optcost = 0;
	int index = 0;

	// Set best trajectory to input trajectory. This will allow us to linearize around the input trajectory for the first iteration.
	StdVectorX Xopt(T);
	StdVectorU Uopt(T-1);
	float deltaopt;

	bool success = true;

	LOG_INFO("delta: %f",delta);
	LOG_INFO("penalty coeff: %f",penalty_coeff);
	LOG_INFO("Trust region size: %.5f", trust_box_size);

	// fill in f. Constant across all iterations because the penalty is constant until we break from this "loop"
	fill_f(penalty_coeff);
	int sqp_iter;
	for(sqp_iter=0; sqp_iter < cfg::max_sqp_iterations; ++sqp_iter) {
		LOG_INFO("  sqp iter: %d",sqp_iter);

		merit = computeMerit(delta, X, U, penalty_coeff, bounds);

		// fill in C, e
		fill_in_C_and_e(X, U, delta, trust_box_size, bounds);
		fill_A_T_and_b_T(X, U, delta, trust_box_size, bounds);

		// if (sqp_iter % 20 == 0) {
		// 	LOG_INFO("		sqp_iter: %d", sqp_iter);
		// }

		for(int trust_iter=0; true; ++trust_iter) {
			LOG_INFO("       trust region size: %f",trust_box_size);

			// fill in lb, ub
			fill_lb_and_ub(X, U, delta, trust_box_size, bounds);

			if (!is_valid_inputs()) {
				LOG_FATAL("ERROR: invalid inputs");
				exit(0);
			}

			// call FORCES
			int exitflag = wam7dofarm_QP_solver_solve(&problem, &output, &info);
			QP_count++;
			if (exitflag == 1) {
				optcost = info.pobj;
				deltaopt = z[0][X_DIM]; // Hard coded, I know the index of this
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
			} else {
				LOG_ERROR("Some problem in QP solver");
				success = false;
				return success;
			}

			float constraint_violation = (computeMerit(deltaopt, Xopt, Uopt, penalty_coeff, bounds) - computeObjective(deltaopt, Xopt, Uopt))/penalty_coeff;
			
			float model_merit = optcost;
			float new_merit = computeMerit(deltaopt, Xopt, Uopt, penalty_coeff, bounds);
			float approx_merit_improve = merit - model_merit;
			float exact_merit_improve = merit - new_merit;
			float merit_improve_ratio = exact_merit_improve/approx_merit_improve;

			LOG_INFO("       approx improve: %.3f. exact improve: %.3f. ratio: %.3f. constraint_violation: %.3f.", approx_merit_improve, exact_merit_improve, merit_improve_ratio, constraint_violation);
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

bool penalty_sqp(StdVectorX& X, StdVectorU& U, float& delta, bounds_t bounds,
		wam7dofarm_QP_solver_params& problem, wam7dofarm_QP_solver_output& output, wam7dofarm_QP_solver_info& info) {
	float penalty_coeff = cfg::initial_penalty_coeff;
	int penalty_increases = 0;

	prev_delta = delta;

	bool success = true;

	while(penalty_increases < cfg::max_penalty_coeff_increases) {
		float constraint_violation = (computeMerit(delta, X, U, penalty_coeff, bounds) - computeObjective(delta, X, U))/penalty_coeff;		
		// std::cout << computeMerit(delta, X, U, penalty_coeff, bounds) << ", " << computeObjective(delta, X, U) << ", " << penalty_coeff <<"\n";
		LOG_INFO("Initial constraint violation: %.5f", constraint_violation);

		success = minimize_merit_function(X, U, delta, bounds, penalty_coeff, problem, output, info);

		if (!success) {
			LOG_ERROR("Merit function not minimized successfully\n");
			break;
		}

		LOG_INFO("Old constraint violation: %.5f", constraint_violation);
		constraint_violation = (computeMerit(delta, X, U, penalty_coeff, bounds) - computeObjective(delta, X, U))/penalty_coeff;
		LOG_INFO("Current constraint violation: %.5f", constraint_violation);
		LOG_INFO("Delta value: %.5f\n", delta);

		if (constraint_violation <= cfg::cnt_tolerance) {
			break;
		} else if (constraint_violation > 10 && penalty_increases == 0) {

			if (delta > bounds.delta_max) {
				LOG_ERROR("Delta exceeds maximum allowed.\n");
				success = false;
				break;
			}

			delta = prev_delta + 0.01;
			//delta = MIN(delta, bounds.delta_max);
			prev_delta = delta;

			Matrix<float, X_DIM, T> init; init.setZero();
			for(int i = 0; i < X_DIM; ++i) {
				init.row(i).setLinSpaced(T, bounds.x_start(i), bounds.x_goal(i));
			}

			for(int t = 0; t < T; ++t) {
				X[t] = init.col(t);
				// X[t] = bounds.x_start;
			}

			// Initialize U variable
			// float c = (bounds.x_goal(3) - bounds.x_start(3)); // Trying this
			for(int t = 0; t < T-1; ++t) {
				U[t] = MatrixXf::Zero(U_DIM+VC_DIM, 1);
				//U[t] = c*MatrixXf::Ones(U_DIM, 1);
			}

			// for(int t = 0; t < T-1; ++t) {
			// 	for(int i = 0; i < U_DIM; ++i) {
			// 		U[t][i] = prev_pi[t*U_DIM + i];
			// 	}
			// }

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
		// for(int t = 0; t < T-1; ++t) {
		//  	X[t+1] = rk45_DP(continuous_dynamics, X[t], U[t], delta, dynamics_weights);
		//  	// std::cout << "X:\n" << X[t+1] << "\n";
		//  	// std::cout << "U:\n" << U[t] << "\n";
		// }
	}
		
	float constraint_violation = (computeMerit(delta, X, U, penalty_coeff, bounds) - computeObjective(delta, X, U))/penalty_coeff;
	LOG_FATAL("Final constraint violation: %.5f", constraint_violation);

	return success;
}

int solve_BVP(float weights[], float pi[], float start_state[], float& delta, float virtual_control_max) {

	// Set pointer to weights
	dynamics_weights = weights;
	prev_pi = pi;

	StdVectorX X(T);
	StdVectorU U(T-1);

	wam7dofarm_QP_solver_params problem;
	wam7dofarm_QP_solver_output output;
	wam7dofarm_QP_solver_info info;
	setup_state_vars(problem, output);

	// Do bounds stuff here.
	// Note that we assume control_u_min = -1*control_u_max, and that it's only one number. This is what I found in Teodor's code
	bounds_t bounds;
	bounds.delta_min = 0.01;
	bounds.delta_max = 0.5;
	bounds.virtual_control_max = virtual_control_max;
	bounds.virtual_control_min = -1 * virtual_control_max;
	vc_max = virtual_control_max;
	for(int i = 0; i < X_DIM; ++i) {
		bounds.x_min(i) = state_min[i]*M_PI/180.0;
		bounds.x_max(i) = state_max[i]*M_PI/180.0;
	}
	for(int i = 0; i < U_DIM; ++i) {
		bounds.u_min(i) = control_min[i]; // control_min and control_max is found in the dynamics file
		bounds.u_max(i) = control_max[i];
	}
	for(int i = 0; i < X_DIM; ++i) {
		bounds.x_start(i) = start_state[i];
		// bounds.x_goal(i) = target_state[i]; // target_state is found in the dynamics file
	}
	for(int i = 0; i < 3; ++i) {
		bounds.pos_goal(i) = target_pos[i];
		bounds.vel_goal(i) = target_vel[i];
	}

	// This is a state that leads to the desired goal position and linear velocity
	float joint_goal[X_DIM] = { 0.        ,  0.        ,  0.        ,  0.        ,  0.       , 0.        ,  0.        ,
								 1.57079633,  1.57079633, -0.78539813,  1.04719753, -3.1415927, 0.78539818,  1.04719757};
	for(int i = 0; i < X_DIM; ++i){
		bounds.x_goal(i) = joint_goal[i];
	}


	// Smart initialization
	//delta = std::min((bounds.x_start - bounds.x_goal).norm()/10, .5);
	delta = 0.1; // 0.01
	//std::cout << "Initial delta: " << delta << "\n";

	// Initialize X variable
	Matrix<float, X_DIM, T> init; init.setZero();
	for(int i = 0; i < X_DIM; ++i) {
		init.row(i).setLinSpaced(T, bounds.x_start(i), bounds.x_goal(i));
	}

	for(int t = 0; t < T; ++t) {
		X[t] = init.col(t);
		// X[t] = bounds.x_start;
	}

	// Initialize U variable
	//float c = (bounds.x_goal(3) - bounds.x_start(3)); // Trying this
	//VectorU umid = bounds.u_min + (bounds.u_max - bounds.u_min)*0.5;
	for(int t = 0; t < T-1; ++t) {
		U[t] = MatrixXf::Zero(U_DIM+VC_DIM, 1);
		//U[t] = umid;
	}

	// Initialize with whatever pi is
	// for(int t = 0; t < T-1; ++t) {
	// 	for(int i = 0; i < U_DIM; ++i) {
	// 		U[t][i] = prev_pi[t*U_DIM + i];
	// 	}
	// }

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

	// std::cout << "Number of QP solves: " << QP_count << "\n";

	// print virtual controls
	// std::cout << "Virtual controls:\n";
	// for(int t = 0; t < T-1; ++t) {
	// 	for(int i = 0; i < VC_DIM; ++i) {
	// 		std::cout << U[t][i+U_DIM] << ", ";
	// 	}
	// 	std::cout << "\n";
	// }

	if (success) {
		std::cout << "States found by SQP:\n";
		std::cout << "[";
		for(int t = 0; t < T; ++t){
			std::cout << "[";
			for(int i = 7; i < X_DIM-1; ++i) {
				std::cout << X[t][i] << ", ";
			}
			std::cout << X[t][X_DIM-1] << "],\n";
		}
		std::cout << "]\n";
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

int main() {

	float weights[70] = {  2.94863503e-01,   7.95022464e-03,   9.31039652e-05,   1.13500168e-01,
   1.87103264e-04,   2.50653430e-01,  -4.77462960e-02,   1.31247765e+00,
  -7.15932785e-03,   1.07676877e+01,   2.60683996e-02,   1.34667720e-05,
   1.17001454e-04,   1.47220170e-02,  -3.65892920e-05,   1.93481366e-02,
  -9.18294328e-03,   1.20340603e-01,   5.97559546e-02,   3.87493756e+00,
   1.36716010e-01,   1.68043419e-02,  -5.09835425e-06,   5.88353541e-03,
   5.29476123e-06,   1.39513702e-01,  -6.89527275e-02,   3.73987270e-01,
   5.96374919e-05,   1.80228141e+00,   5.71926891e-02,  -1.46649609e-05,
  -8.19310407e-05,   5.71647103e-02,   9.41699492e-05,   3.00440392e-03,
   1.19651257e-02,  -5.50646552e-04,   3.18542190e-01,   2.40016804e+00,
   5.58751164e-05,  -2.56441662e-07,   1.88221258e-09,   7.81717177e-05,
   8.32615073e-07,   6.59353687e-05,   1.10406465e-05,   6.32683131e-04,
   5.39376610e-04,   1.23760190e-01,   9.31066784e-04,  -1.48291976e-06,
   2.00506978e-06,   4.98334357e-04,   2.21618420e-04,   5.74834996e-04,
  -5.12519277e-05,  -7.11890196e-03,   1.03169938e-02,   4.17973640e-01,
   3.84504696e-05,   1.90892966e-07,  -1.77110878e-08,   3.87790793e-05,
   3.62328787e-08,   7.40822633e-05,  -5.47395404e-06,   1.11984716e-05,
  -2.22110456e-04,   6.86475300e-02};

	// Start and goal state
	//VectorX x_start, x_goal;

	//float initial_state[NX] = { 0.0, 0.0, -0.00116139991132, 0.312921810594 };
	//float initial_state[NX] = { 0.0, 0.0, 0.010977943328, 0.120180320014 };

	//x_start << 0.0, 0.0, -0.00116139991132, 0.312921810594;
	//x_start << 0.0, 0.0, 3.0, 0.0;
	//x_goal << 0.0, 0.0, 3.14159265359, 0.0;


	//float start_state[X_DIM] = {0.0, 0.0, -0.00116139991132, 0.312921810594};
	//float start_state[X_DIM] = {0.0, 0.0, 3.0, 0.0};
	//float start_state[X_DIM] = {0.0, 0.0, 0.0, -100.0};
	float start_state[X_DIM] = {-0.01508158,  0.01070279,  0.0741887 , -0.11166568,  0.01756522,
       -0.11350337, -0.05654618, -0.02254208,  0.01310971, -0.04482627,
       -0.0614651 ,  0.11027468,  0.09555692, -0.19752187};
	//float start_state[X_DIM] = {0.0, 0.0, 0.0, 0.312921810594};
	float pi[(T-1)*U_DIM];
	float delta = 1;

	float model_slack_bounds = 5.0;

	// Time it
	std::clock_t start;
	start = std::clock();

	bool success = solve_BVP(weights, pi, start_state, delta, model_slack_bounds);

	float solvetime = ( std::clock() - start ) / (float) CLOCKS_PER_SEC;

	printf("Solve time: %5.3f s\n", solvetime);

	if (success) {
		printf("Success!!\n");
		// VectorXf PI(U_DIM);
		// for (int i = 0; i < U_DIM; i++) {
		// 	PI(i) = pi[i];
		// }
		// printf("pi:\n");
		// std::cout << PI << "\n";
	} else {
		printf("Failure...\n");
		// VectorXf PI(U_DIM);
		// for (int i = 0; i < U_DIM; i++) {
		// 	PI(i) = pi[i];
		// }
		// printf("pi:\n");
		// std::cout << PI << "\n";
	}

	cleanup_state_vars();
}

