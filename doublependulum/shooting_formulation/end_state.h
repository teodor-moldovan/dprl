
#ifndef END_STATE_H_
#define END_STATE_H_

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#define NX 4    // size of state space 
#define NU 2    // number of controls
#define NV 4    // number of virtual controls
#define NT 4    // number of constrained target state components
#define NW 72    // number of weights (dynamics parameters)

#define EPS 1e-5

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/LU>
using namespace Eigen;        

typedef Matrix<double, NX, 1> VectorX;
typedef Matrix<double, NU+NV, 1> VectorU;
typedef std::vector<VectorU> StdVectorU;

#include "../doublependulum_dynamics_by_hand.h"
using namespace doublependulum;

double uniform(double low, double high) {
  return (high - low)*(rand() / double(RAND_MAX)) + low;
}

VectorX end_state(VectorX& x_start, StdVectorU& U, double delta, double weights[]) {

  VectorX temp = x_start;
  for(int t = 0; t < (int) U.size(); ++t) {
    temp = rk45_DP(continuous_dynamics, temp, U[t], delta, weights);
    // std::cout << "Intermediate state " << t+1 << ":\n";
    // std::cout << temp << std::endl;
  }
  return temp;

}

double goal_cost(VectorX& x_goal, VectorX& x_start, StdVectorU& U, double delta, double weights[]) {

  VectorX propagated_end = end_state(x_start, U, delta, weights);
  return (x_goal - propagated_end).transpose() * (x_goal - propagated_end);

}

MatrixXd end_state_numerical_jacobian(VectorX& x_start, StdVectorU& U, double delta, double weights[])
{
    int total_controls = U.size()*(NU+NV);
    int nX = x_start.rows();

    // Create matrix, set it to all zeros
    MatrixXd jac(nX, total_controls+1);
    jac.setZero(nX, total_controls+1);

    int index = 0;

    jac.col(index) = end_state(x_start, U, delta+.5*EPS, weights) - end_state(x_start, U, delta - .5*EPS, weights);

    StdVectorU forward_time(U);
    StdVectorU backward_time(U);
    for(int t = 0; t < (int) U.size(); ++t) {
      for(int i = 0; i < NU+NV; ++i) {
          double f_cache = forward_time[t][i]; double b_cache = backward_time[t][i];
          forward_time[t][i] += .5*EPS;
          backward_time[t][i] -= .5*EPS;          
          jac.col(index) = end_state(x_start, forward_time, delta, weights) - end_state(x_start, backward_time, delta, weights);
          index++;
          forward_time[t][i] = f_cache;
          backward_time[t][i] = b_cache; 
      }
    }


    // Must divide by eps for finite differences formula
    jac /= EPS;

    return jac;
}

VectorXd goal_cost_numerical_gradient(VectorX& x_goal, VectorX& x_start, StdVectorU& U, double delta, double weights[]) {

    int total_controls = U.size()*(NU+NV+1);

    // Create matrix, set it to all zeros
    VectorXd grad(total_controls);
    grad.setZero();

    int index = 0;
    double grad_delta = goal_cost(x_goal, x_start, U, delta+.5*EPS, weights) - goal_cost(x_goal, x_start, U, delta - .5*EPS, weights);

    StdVectorU forward_time(U);
    StdVectorU backward_time(U);
    for(int t = 0; t < (int) U.size(); ++t) {
      grad(index) = grad_delta;
      index++;
      for(int i = 0; i < NU+NV; ++i) {
          double f_cache = forward_time[t][i]; double b_cache = backward_time[t][i];
          forward_time[t][i] += .5*EPS;
          backward_time[t][i] -= .5*EPS;          
          grad(index) = goal_cost(x_goal, x_start, forward_time, delta, weights) - goal_cost(x_goal, x_start, backward_time, delta, weights);
          index++;
          forward_time[t][i] = f_cache;
          backward_time[t][i] = b_cache; 
      }
    }

    // Must divide by eps for finite differences formula
    grad /= EPS;

    return grad;

}

#endif /* END_STATE_H_ */

            