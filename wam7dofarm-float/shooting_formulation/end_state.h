
#ifndef END_STATE_H_
#define END_STATE_H_

#include <iostream>
#include <vector>
#include <math.h>
#define NX 14    // size of state space 
#define NU 7    // number of controls
#define NV 14    // number of virtual controls
#define NT 6    // number of constrained target state components

#define EPS 1e-5

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/LU>
using namespace Eigen;        

typedef Matrix<double, NX, 1> VectorX;
typedef Matrix<double, NU+NV, 1> VectorU;
typedef std::vector<VectorU> StdVectorU;

#include "../wam7dofarm_dynamics_by_hand.h"
using namespace wam7dofarm;


VectorXd end_state(StdVectorU U, VectorX x_start, double delta, double weights[]) {

  VectorX temp = x_start;
  for(int t = 0; t < U.size(); ++t) {
    temp = rk45_DP(continuous_dynamics, temp, U[t], delta, weights);
    // std::cout << "Intermediate state " << t+1 << ":\n";
    // std::cout << temp << std::endl;
  }
  return temp;

}

MatrixXd end_state_numerical_jacobian(VectorX x_start, StdVectorU U, double delta, double weights[])
{
    int total_controls = U.size()*(NU+NV);
    int nX = x_start.rows();

    // Create matrix, set it to all zeros
    MatrixXd jac(nX, total_controls+1);
    jac.setZero(nX, total_controls+1);

    int index = 0;

    StdVectorU forward_time(U);
    StdVectorU backward_time(U);
    for(int t = 0; t < U.size(); ++t) {
      for(int i = 0; i < NU+NV; ++i) {
          double f_cache = forward_time[t][i]; double b_cache = backward_time[t][i];
          forward_time[t][i] += .5*EPS;
          backward_time[t][i] -= .5*EPS;          
          jac.col(index) = end_state(forward_time, x_start, delta, weights) - end_state(backward_time, x_start, delta, weights);
          index++;
          forward_time[t][i] = f_cache;
          backward_time[t][i] = b_cache; 
      }
    }

    jac.col(index) = end_state(U, x_start, delta+.5*EPS, weights) - end_state(U, x_start, delta - .5*EPS, weights);

    // Must divide by eps for finite differences formula
    jac /= EPS;

    return jac;
}

#endif /* END_STATE_H_ */

            