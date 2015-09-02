
#ifndef CARTPOLE_DYNAMICS_BY_HAND_H_
#define CARTPOLE_DYNAMICS_BY_HAND_H_

#include <iostream>
#include <math.h>
#define NX 4    // size of state space 
#define NU 1    // number of controls
#define NV 4    // number of virtual controls
#define NT 4    // number of constrained target state components
#define NW 60    // number of weights (dynamics parameters)

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/LU>
using namespace Eigen;        

double control_min[NU] = { -1.0 };
double control_max[NU] = { 1.0 };

#define EPS 1e-5
        
// Target state may only be specified partially. 
// Indices of target state components that are constrained
int target_ind[NT] = { 0, 1, 2, 3 };
// Constraint values
double target_state[NT] = { 0.0, 0.0, 3.14159265359, 0.0 };

                 

namespace cartpole {

    VectorXd rk45_DP(VectorXd (*f)(VectorXd, VectorXd, double[]), VectorXd x, VectorXd u, double delta, double weights[]) 
    // void rk45_DP(float f(float, float), float* t, float* x, float h, float* epsilon)
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

    VectorXd rk45(VectorXd (*f)(VectorXd, VectorXd, double[]), VectorXd x, VectorXd u, double delta, double weights[]) 
    // void rk45_DP(float f(float, float), float* t, float* x, float h, float* epsilon)
    {

      float c20 = 0.25, c21 = 0.25;
      float c30 = 0.375, c31 = 0.09375, c32 = 0.28125;
      float c40,c41, c42,c43;
      float c51, c52 = -8.0, c53, c54;
      float c60 = 0.5, c61, c62 = 2, c63, c64;
      float c65 = -0.275;
      float a1, a2 = 0, a3, a4, a5 = -0.2;
      float b1, b2 = 0, b3, b4, b5= -0.18, b6;
      VectorXd F1, F2, F3, F4, F5, F6, x4, x_new;

      c40 = (float) 12/ (float) 13;
      c41 = (float) 1932/(float) 2197;
      c42 = (float) -7200/(float) 2197;
      c43 = (float) 7296/(float) 2197;
      c51 = c53 = (float) 439/ (float) 216;
      c54 = (float) -845/(float) 4104;
      c61 = (float) -8/(float) 27;
      c63 = (float) -3544/(float) 2565;
      c64 = (float) 1859/(float) 4104;
      a1 = (float) 25/(float) 216;
      a3 = (float) 1408/(float) 2565;
      a4 = (float) 2197/(float) 4104;
      b1 = (float) 16/(float) 135;
      b3 = (float) 6656/(float) 12825;
      b4 = (float) 28561/(float) 56430;
      b6 = (float) 2/(float) 55;


      F1 = delta * f(x, u, weights);
      F2 = delta * f(x + c21 * F1, u, weights);
      F3 = delta * f(x + c31 * F1 + c32 * F2, u, weights);
      F4 = delta * f(x + c41 * F1 + c42 * F2 + c43 * F3, u, weights);
      F5 = delta * f(x + c51 * F1 + c52 * F2 + c53 * F3 + c54 * F4, u, weights);
      F6 = delta * f(x + c61 * F1 + c62 * F2 + c63 * F3 + c64 * F4 + c65 * F5, u, weights);

      x4 = x + a1 * F1 + a3 * F3 + a4 * F4 + a5 * F5;
      x_new = x + b1 * F1 + b3 * F3 + b4 * F4 + b5 * F5 + b6 * F6;
      // double epsilon = (x_new - x4).norm();

      return x_new;

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

    VectorXd continuous_dynamics(VectorXd x, VectorXd u, double weights[])
    {

        // State is: [w, v, t, x] = [dt, dx, t, x]
        VectorXd xdot(NX);

        double max_control = 10.0;

        // Squashing
        for (int i = 0; i < NU; ++i) {
            u(i) = (1.0 / (1.0 + exp(-1.0 * u(i))) - 0.5) * 2 * max_control;
        }

        // Hand coded dynamics for cartpole using weights
        Matrix<double, 2, 2> A;
        Matrix<double, 2, 1> b;
        A.setZero(); b.setZero();

        A << weights[1]*cos(x(2)), weights[0],
             weights[5], weights[4]*cos(x(2));

        double g = 9.82; // Gravity
        b << u(0) - weights[2]*x(0)*x(0)*sin(x(2)) - weights[3]*x(1) + u(NU+0),
             g*sin(x(2)) + u(NU+1);

        Vector2d temp = A.inverse() * b;

        xdot(0) = temp(0);
        xdot(1) = temp(1);
        xdot(2) = x(0) + u(NU+2);
        xdot(3) = x(1) + u(NU+3);        

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

    MatrixXd numpy_array_to_Eigen_matrix(double matrix[], int rows, int cols)
    {

      MatrixXd m(rows, cols);

      for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
          m(i,j) = matrix[i*cols + j];
        }
      }

      // std::cout << "Matrix:\n" << m << std::endl;

      return m;

    }

    void Eigen_matrix_to_numpy_array(MatrixXd matrix, double* m)
    {

      int rows = matrix.rows();
      int cols = matrix.cols();

      // Put it in row major..?
      for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
          m[i*cols + j] = matrix(i,j);
        }
      }

    }

    void integrate_forward(double* x_, double* u_, double delta, double* weights, double* x_next)
    {

      // Input conversion
      VectorXd x(NX); x = numpy_array_to_Eigen_matrix(x_, NX, 1);
      VectorXd u(NU+NV); u = numpy_array_to_Eigen_matrix(u_, NU+NV, 1);

      VectorXd x_next_(NX); x_next_ = rk45_DP(continuous_dynamics, x, u, delta, weights);

      // std::cout << "x_next:\n" << x_next_ << "\n";

      // Output conversion
      for(int i = 0; i < NX; ++i) {
        x_next[i] = x_next_(i);
      }
    }

    void linearize_dynamics(double* x_, double* u_, double delta, double* weights, double* jac_x, double* jac_u)
    {

      // Input conversion
      VectorXd x(NX); x = numpy_array_to_Eigen_matrix(x_, NX, 1);
      VectorXd u(NU+NV); u = numpy_array_to_Eigen_matrix(u_, NU+NV, 1);

      Matrix<double, NX, NX+NU+NV+1> jac = numerical_jacobian(continuous_dynamics, x, u, delta, weights);
      Matrix<double, NX, NX> DH_X = jac.leftCols(NX);
      Matrix<double, NX, NU+NV> DH_U = jac.middleCols(NX, NU+NV);

      // std::cout << "DH_X:\n" << DH_X << "\n";
      // std::cout << "DH_U:\n" << DH_U << "\n";

      // Output conversion
      Eigen_matrix_to_numpy_array(DH_X, jac_x);
      Eigen_matrix_to_numpy_array(DH_U, jac_u);

    }



};

#endif /* CARTPOLE_DYNAMICS_BY_HAND_H_ */

            
