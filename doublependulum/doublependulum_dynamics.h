
#ifndef DOUBLEPENDULUM_DYNAMICS_H_
#define DOUBLEPENDULUM_DYNAMICS_H_


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

double true_weights[NW] = { 0.0, 0.0, -0.166666666667, 0.0, 0.0, -0.0625, 0.0, -0.0625, 2.0, 3.6825, 0.0, 0.0, 0.0, -0.0625, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.0, 0.0, -0.0416666666667, -0.0625, 0.0, -0.0625, 0.0, 0.0, 0.0, 0.0, -0.0625, 1.2275, 0.0, 2.0, 0.0625, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };

namespace doublependulum {

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
    double tmp10;
    double tmp11;
    
    tmp0 =  sin(z[2]);
    tmp1 =  sin(z[3]);
    tmp2 =  tmp0*tmp1;
    tmp3 =  cos(z[2]);
    tmp4 =  cos(z[3]);
    tmp5 =  tmp3*tmp4;
    tmp6 =  pow(z[0], 2);
    tmp7 =  tmp1*tmp3*tmp6;
    tmp8 =  pow(z[1], 2);
    tmp9 =  tmp0*tmp4*tmp8;
    tmp10 =  tmp0*tmp4*tmp6;
    tmp11 =  tmp1*tmp3*tmp8;
    out[0] =  tmp2*weights[4] + tmp5*weights[6] + weights[2];
    out[1] =  tmp2*weights[5] + tmp5*weights[7] + weights[3];
    out[2] =  weights[0];
    out[3] =  weights[1];
    out[4] =  tmp0*weights[9] + tmp1*weights[12] + tmp10*weights[15] + tmp11*weights[16] + tmp7*weights[11] + tmp9*weights[13] + weights[10]*z[0] + weights[14]*z[5] + weights[17]*z[1] + weights[8]*z[4] + z[6];
    out[5] =  tmp2*weights[22] + tmp5*weights[24] + weights[20];
    out[6] =  tmp2*weights[23] + tmp5*weights[25] + weights[21];
    out[7] =  weights[18];
    out[8] =  weights[19];
    out[9] =  tmp0*weights[27] + tmp1*weights[30] + tmp10*weights[33] + tmp11*weights[34] + tmp7*weights[29] + tmp9*weights[31] + weights[26]*z[4] + weights[28]*z[0] + weights[32]*z[5] + weights[35]*z[1] + z[7];
    out[10] =  tmp2*weights[40] + tmp5*weights[42] + weights[38];
    out[11] =  tmp2*weights[41] + tmp5*weights[43] + weights[39];
    out[12] =  weights[36];
    out[13] =  weights[37];
    out[14] =  tmp0*weights[45] + tmp1*weights[48] + tmp10*weights[51] + tmp11*weights[52] + tmp7*weights[47] + tmp9*weights[49] + weights[44]*z[4] + weights[46]*z[0] + weights[50]*z[5] + weights[53]*z[1] + z[8];
    out[15] =  tmp2*weights[58] + tmp5*weights[60] + weights[56];
    out[16] =  tmp2*weights[59] + tmp5*weights[61] + weights[57];
    out[17] =  weights[54];
    out[18] =  weights[55];
    out[19] =  tmp0*weights[63] + tmp1*weights[66] + tmp10*weights[69] + tmp11*weights[70] + tmp7*weights[65] + tmp9*weights[67] + weights[62]*z[4] + weights[64]*z[0] + weights[68]*z[5] + weights[71]*z[1] + z[9];
    
    
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

#endif /* DOUBLEPENDULUM_DYNAMICS_H_ */