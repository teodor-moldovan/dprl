
#ifndef PENDULUM_DYNAMICS_BY_HAND_H_
#define PENDULUM_DYNAMICS_BY_HAND_H_

#include <iostream>
#include <math.h>
#define NX 2    // size of state space 
#define NU 1    // number of controls
#define NV 2    // number of virtual controls
#define NT 2    // number of constrained target state components
#define NW 10    // number of weights (dynamics parameters)

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/LU>
using namespace Eigen;        

double control_min[NU] = { -1.0 };
double control_max[NU] = { 1.0 };

#define EPS 1e-5

// Target state may only be specified partially. 
// Indices of target state components that are constrained
int target_ind[NT] = { 0, 1 };
// Constraint values
double target_state[NT] = { 0.0, 3.14159265359 };

namespace pendulum {

    VectorXd rk45_DP(VectorXd (*f)(VectorXd, VectorXd, double[]), VectorXd x, VectorXd u, double delta, double weights[]) 
    // void rk45_DP5_DP(float f(float, float), float* t, float* x, float h, float* epsilon)
    {

      float c21;
      float c31, c32;
      float c41, c42, c43;
      float c51, c52, c53, c54;
      float c61, c62, c63, c64, c65;
      float c71, c72, c73, c74, c75, c76;
      float a1, a2 = 0, a3, a4, a5, a6, a7 = 0;
      float b1, b2 = 0, b3, b4, b5, b6, b7;
      VectorXd F1, F2, F3, F4, F5, F6, F7, y, z;

      c21 = (float) 1/(float) 5;
      c31 = (float) 3/(float) 40;
      c32 = (float) 9/(float) 40;
      c41 = (float) 44/(float) 45;
      c42 = (float) -56/(float) 15;
      c43 = (float) 32/(float) 9;
      c51 = (float) 19372/(float) 6561;
      c52 = (float) -25360/(float) 2187;
      c53 = (float) 64448/(float) 6561;
      c54 = (float) -212/(float) 729;
      c61 = (float) 9017/(float) 3168;
      c62 = (float) -355/(float) 33;
      c63 = (float) 46732/(float) 5247;
      c64 = (float) 49/(float) 176;
      c65 = (float) -5103/(float) 18656;
      c71 = (float) 35/(float) 384;
      c72 = (float) 0;
      c73 = (float) 500/(float) 1113;
      c74 = (float) 125/(float) 192;
      c75 = (float) -2187/(float) 6784;
      c76 = (float) 11/(float) 84;
      a1 = (float) 35/(float) 384;
      a3 = (float) 500/(float) 1113;
      a4 = (float) 125/(float) 192;
      a5 = (float) -2187/(float) 6784;
      a6 = (float) 11/(float) 84;
      b1 = (float) 5179/(float) 57600;
      b3 = (float) 7571/(float) 16695;
      b4 = (float) 393/(float) 640;
      b5 = (float) -92097/(float) 339200;
      b6 = (float) 187/(float) 2100;
      b7 = (float) 1/(float) 40;

      F1 = delta * f(x, u, weights);
      F2 = delta * f(x + c21 * F1, u, weights);
      F3 = delta * f(x + c31 * F1 + c32 * F2, u, weights);
      F4 = delta * f(x + c41 * F1 + c42 * F2 + c43 * F3, u, weights);
      F5 = delta * f(x + c51 * F1 + c52 * F2 + c53 * F3 + c54 * F4, u, weights);
      F6 = delta * f(x + c61 * F1 + c62 * F2 + c63 * F3 + c64 * F4 + c65 * F5, u, weights);
      F7 = delta * f(x + c71 * F1 + c72 * F2 + c73 * F3 + c74 * F4 + c75 * F5 + c76 * F6, u, weights);

      y = x + a1 * F1 + a2 * F2 + a3 * F3 + a4 * F4 + a5 * F5 + a6 * F6 + a7 * F7; // fifth order accuracy
      z = x + b1 * F1 + b2 * F2 + b3 * F3 + b4 * F4 + b5 * F5 + b6 * F6 + b7 * F7; // fourth order accuracy
      double epsilon = (y - z).norm();
      // std::cout << "Error: " << epsilon << ", Delta: " << delta << "\n";

      return y;

    }

    VectorXd rk4(VectorXd (*f)(VectorXd, VectorXd, double[]), VectorXd x, VectorXd u, double delta, double weights[]) 
    {
        VectorXd k1 = delta*f(x, u, weights);
        VectorXd k2 = delta*f(x + .5*k1, u, weights);
        VectorXd k3 = delta*f(x + .5*k2, u, weights);
        VectorXd k4 = delta*f(x + k3, u, weights);

        VectorXd x_new = x + (k1 + 2*k2 + 2*k3 + k4)/6;
        return x_new;
    }

    void M( double* M_out, double* parms )
    {
      M_out[0] = parms[5];
    } 

    void c( double* c_out, double* parms )
    {
      c_out[0] = 0;
    }

    void g( double* g_out, double* parms, double* q )
    {
      g_out[0] = 9.81*parms[6]*cos(q[0]) - 9.81*parms[7]*sin(q[0]);
    } 

    VectorXd continuous_dynamics(VectorXd x, VectorXd u, double weights[])
    {

        const int half_NX = NX/2;
        double max_control = 5.0;

        // Create Eigen data structures
        Matrix<double, half_NX, 1> tau;
        Matrix<double, NV, 1> virtual_control;
        Matrix<double, half_NX, 1> c_vec;
        Matrix<double, half_NX, 1> g_vec;
        Matrix<double, half_NX, half_NX> M_mat;
        Matrix<double, half_NX, 1> ddq; // For fully actuated system: NU = half_NX
        Matrix<double, half_NX, 1> _ddq;

        // Create local arrays to pass into generated code
        double M_out[half_NX], g_out[half_NX], c_out[half_NX], q[half_NX], dq[half_NX];

        // Fill in q and dq
        int idx = 0;
        for(int i = 0; i < half_NX; ++i) {
          dq[i] = x(idx++);
        }
        for(int i = 0; i < half_NX; ++i) {
          q[i] = x(idx++);
        }

        // Calculate using generated code
        M(M_out, weights);
        c(c_out, weights);
        g(g_out, weights, q);

        // Fill in vectors and matrices
        for(int i = 0; i < half_NX; ++i) {
          c_vec(i) = c_out[i];
          g_vec(i) = g_out[i];
        }

        idx = 0;
        for(int i = 0; i < half_NX; ++i) {
          for(int j = 0; j < half_NX; ++j) {
            M_mat(i,j) = M_out[idx++]; // Row major
          }
        }

        tau = u.head(NU);
        virtual_control = u.tail(NV);

        // Now compute ddq
        ddq = M_mat.lu().solve(max_control*tau - c_vec - g_vec + virtual_control.head(half_NX));
        // ddq = M_mat.inverse() * (tau - c_vec - g_vec);

        // Stack [ddq, dq] into xdot
        Vector2d xdot;
        xdot.head(half_NX) << ddq;
        xdot.tail(half_NX) << x.head(half_NX) + virtual_control.tail(half_NX); // first half of x is dq

        return xdot;
    }

    MatrixXd numerical_jacobian(VectorXd (*f)(VectorXd, VectorXd, double[]), VectorXd x, VectorXd u, double delta, double weights[])
    {
        int nX = x.rows();
        int nU = u.rows();

        // Create matrix, set it to all zeros
        MatrixXd jac(nX, nX+nU+1);
        jac.setZero(nX, nX+nU+1);

        int index = 0;

        MatrixXd I;
        I.setIdentity(nX, nX);
        for(int i = 0; i < nX; ++i) {
            jac.col(index) = rk45_DP(f, x + .5*EPS*I.col(i), u, delta, weights) - rk45_DP(f, x - .5*EPS*I.col(i), u, delta, weights);
            index++;
        }

        I.setIdentity(nU, nU);
        for(int i = 0; i < nU; ++i) {
            jac.col(index) = rk45_DP(f, x, u + .5*EPS*I.col(i), delta, weights) - rk45_DP(f, x, u - .5*EPS*I.col(i), delta, weights);
            index++;
        }

        jac.col(index) = rk45_DP(f, x, u, delta + .5*EPS, weights) - rk45_DP(f, x, u, delta - .5*EPS, weights);

        // Must divide by eps for finite differences formula
        jac /= EPS;

        return jac;
    }

    VectorXd dynamics_difference(VectorXd (*f)(VectorXd, VectorXd, double[]), VectorXd x, VectorXd x_next, VectorXd u, double delta, double weights[])
    {
        VectorXd simulated_x_next = rk45_DP(f, x, u, delta, weights);
        return x_next - simulated_x_next;
    }

};

#endif /* PENDULUM_DYNAMICS_BY_HAND_H_ */

            