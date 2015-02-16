
#ifndef CARTPOLE_DYNAMICS_H_
#define CARTPOLE_DYNAMICS_H_


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

double true_weights[NW] = { 0.0, -1.625, 0.0, 0.0, 0.375, 0.0, 0.0, -58.92, 0.0, 0.0, -0.375, -60.0, 0.6, 0.0, 0.0, -3.25, 0.0, 0.0, 0.0, 0.0, 0.75, -0.4, 0.0, 40.0, 7.365, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

namespace cartpole {

    /* Function specifying dynamics.
    Format assumed: M(x) * \dot_x + g(x,u) = 0
    where x is state and u are controls

    Input:
        z : [x,u,xi] concatenated; z is assumed to have dimension NX + NU + NV
        weights : weight parameter vector as provided by learning component 
    Output: 
        M : output array equal to [M,g] concatenated,
              an NX x (NX + 1) matrix flattened in row major order.
              (note this output array needs to be pre-allocated)
        
    */

    void feval(double z[], double weights[], double out[])
    { 
        
    
    
    double tmp0;
    double tmp1;
    double tmp2;
    double tmp3;
    double tmp4;
    double tmp5;
    double tmp6;
    double tmp7;
    double tmp8;
    double tmp9;
    
    tmp0 =  2*z[2];
    tmp1 =  cos(tmp0);
    tmp2 =  sin(z[2]);
    tmp3 =  cos(z[2]);
    tmp4 =  tmp3*z[4];
    tmp5 =  tmp3*z[1];
    tmp6 =  sin(tmp0);
    tmp7 =  pow(z[0], 2);
    tmp8 =  tmp2*tmp7;
    tmp9 =  tmp6*tmp7;
    out[0] =  tmp1*weights[4] + weights[1];
    out[1] =  tmp1*weights[5] + weights[0];
    out[2] =  weights[2];
    out[3] =  weights[3];
    out[4] =  tmp2*weights[7] + tmp4*weights[11] + tmp5*weights[12] + tmp6*weights[9] + tmp8*weights[13] + tmp9*weights[10] + weights[14]*z[0] + weights[6]*z[1] + weights[8]*z[4] + z[5];
    out[5] =  tmp1*weights[19] + weights[16];
    out[6] =  tmp1*weights[20] + weights[15];
    out[7] =  weights[17];
    out[8] =  weights[18];
    out[9] =  tmp2*weights[22] + tmp4*weights[26] + tmp5*weights[27] + tmp6*weights[24] + tmp8*weights[28] + tmp9*weights[25] + weights[21]*z[1] + weights[23]*z[4] + weights[29]*z[0] + z[6];
    out[10] =  tmp1*weights[34] + weights[31];
    out[11] =  tmp1*weights[35] + weights[30];
    out[12] =  weights[32];
    out[13] =  weights[33];
    out[14] =  tmp2*weights[37] + tmp4*weights[41] + tmp5*weights[42] + tmp6*weights[39] + tmp8*weights[43] + tmp9*weights[40] + weights[36]*z[1] + weights[38]*z[4] + weights[44]*z[0] + z[7];
    out[15] =  tmp1*weights[49] + weights[46];
    out[16] =  tmp1*weights[50] + weights[45];
    out[17] =  weights[47];
    out[18] =  weights[48];
    out[19] =  tmp2*weights[52] + tmp4*weights[56] + tmp5*weights[57] + tmp6*weights[54] + tmp8*weights[58] + tmp9*weights[55] + weights[51]*z[1] + weights[53]*z[4] + weights[59]*z[0] + z[8];
    
    
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
        double Mg[NX*(NX+1)];
        double z[NX+NU+NV];

        // state
        for(int i = 0; i < NX; ++i) {
            z[i] = x(i);
        }
        // controls
        for(int i = 0; i < NU; ++i) {
            z[i+NX] = u(i);
        }
        // virtual controls
        for(int i = 0; i < NV; ++i) {
            z[i+NX+NU] = u(i+NU);
        }

        //z[0] = x(0); z[1] = x(1); z[2] = x(2); z[3] = x(3); z[4] = u(0);

        feval(z, weights, Mg);

        MatrixXd M(NX,NX);
        VectorXd g(NX);

        int idx = 0;
        for(int i = 0; i < NX; ++i) {
            for(int j = 0; j < NX; ++j) {
                M(i,j) = Mg[idx++];
            }
            g(i) = Mg[idx++];
        }

        VectorXd xdot(NX);
        xdot = M.lu().solve(-g);

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

#endif /* CARTPOLE_DYNAMICS_H_ */

            