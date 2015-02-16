
#ifndef DOUBLEPENDULUM_DYNAMICS_BY_HAND_H_
#define DOUBLEPENDULUM_DYNAMICS_BY_HAND_H_


#include <math.h>
#define NX 4    // size of state space 
#define NU 2    // number of controls
#define NV 4    // number of virtual controls
#define NT 4    // number of constrained target state components
#define NW 72    // number of weights (dynamics parameters)

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/LU>
using namespace Eigen;        

double control_min[NU] = { -1.0, -1.0 };
double control_max[NU] = { 1.0, 1.0 };

#define EPS 1e-5
        
// Target state may only be specified partially. 
// Indices of target state components that are constrained
int target_ind[NT] = { 0, 1, 2, 3 };
// Constraint values
double target_state[NT] = { 0.0, 0.0, 0.0, 0.0 };

                 

namespace doublependulum {

    
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
        
        // State is: [w1, w2, t1, t2]
        VectorXd xdot(NX);      

        // Hand coded dynamics for doublependulum using weights
        Matrix<double, 2, 2> A;
        Matrix<double, 2, 1> b;
        A.setZero(); b.setZero();

        A << weights[0], weights[1] * cos(x(2)-x(3)),
             weights[5] * cos(x(2)-x(3)), weights[6];

        b << 2*u(0) - weights[2]*x(1)*x(1)*sin(x(2)-x(3)) - weights[3]*sin(x(2)) - weights[4]*x(0)+ u(NU+0),
             2*u(1) - weights[7]*x(0)*x(0)*sin(x(2)-x(3)) - weights[8]*sin(x(3)) - weights[9]*x(1)+ u(NU+1);

        Vector2d temp = A.inverse() * b;
        
        xdot(0) = temp(0) ;
        xdot(1) = temp(1) ;
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
            jac.col(index) = rk4(f, x + .5*EPS*I.col(i), u, delta, weights) - rk4(f, x - .5*EPS*I.col(i), u, delta, weights);
            index++;
        }

        I.setIdentity(nU, nU);
        for(int i = 0; i < nU; ++i) {
            jac.col(index) = rk4(f, x, u + .5*EPS*I.col(i), delta, weights) - rk4(f, x, u - .5*EPS*I.col(i), delta, weights);
            index++;
        }

        jac.col(index) = rk4(f, x, u, delta + .5*EPS, weights) - rk4(f, x, u, delta - .5*EPS, weights);

        // Must divide by eps for finite differences formula
        jac /= EPS;

        return jac;
    }

    VectorXd dynamics_difference(VectorXd (*f)(VectorXd, VectorXd, double[]), VectorXd x, VectorXd x_next, VectorXd u, double delta, double weights[])
    {
        VectorXd simulated_x_next = rk4(f, x, u, delta, weights);
        return x_next - simulated_x_next;
    }

};

#endif /* DOUBLEPENDULUM_DYNAMICS_BY_HAND_H_ */

            