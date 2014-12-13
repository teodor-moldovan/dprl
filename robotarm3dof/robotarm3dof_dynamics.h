
#ifndef ROBOTARM3DOF_DYNAMICS_H_
#define ROBOTARM3DOF_DYNAMICS_H_


#include <math.h>
#define NX 6    // size of state space 
#define NU 3    // number of controls
#define NV 6    // number of virtual controls
#define NT 6    // number of constrained target state components
#define NW 252    // number of weights (dynamics parameters)

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/LU>
using namespace Eigen;        

double control_min[NU] = { -1.0, -1.0, -1.0 };
double control_max[NU] = { 1.0, 1.0, 1.0 };

#define EPS 1e-5
        
// Target state may only be specified partially. 
// Indices of target state components that are constrained
int target_ind[NT] = { 0, 1, 2, 3, 4, 5 };
// Constraint values
double target_state[NT] = { 0.785398163397, -0.785398163397, 0.785398163397, 0.0, 0.0, 0.0 };

                 

namespace robotarm3dof {

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
    double tmp12;
    double tmp13;
    double tmp14;
    double tmp15;
    double tmp16;
    double tmp17;
    double tmp18;
    double tmp19;
    double tmp20;
    double tmp21;
    double tmp22;
    double tmp23;
    double tmp24;
    double tmp25;
    double tmp26;
    double tmp27;
    double tmp28;
    double tmp29;
    double tmp30;
    double tmp31;
    double tmp32;
    double tmp33;
    double tmp34;
    
    tmp0 =  cos(z[2]);
    tmp1 =  2*z[1];
    tmp2 =  cos(tmp1);
    tmp3 =  sin(z[2]);
    tmp4 =  sin(tmp1);
    tmp5 =  tmp3*tmp4;
    tmp6 =  tmp0*tmp2;
    tmp7 =  2*z[2];
    tmp8 =  sin(tmp7);
    tmp9 =  tmp4*tmp8;
    tmp10 =  cos(tmp7);
    tmp11 =  tmp10*tmp2;
    tmp12 =  cos(z[1]);
    tmp13 =  tmp0*tmp12;
    tmp14 =  tmp3*sin(z[1]);
    tmp15 =  tmp3*z[3]*z[5];
    tmp16 =  tmp3*z[4]*z[5];
    tmp17 =  tmp3*pow(z[5], 2);
    tmp18 =  tmp3*pow(z[4], 2);
    tmp19 =  pow(z[3], 2);
    tmp20 =  tmp19*tmp3;
    tmp21 =  tmp4*z[3]*z[4];
    tmp22 =  tmp19*tmp4;
    tmp23 =  tmp2*tmp3*z[3]*z[5];
    tmp24 =  tmp2*tmp3*z[3]*z[4];
    tmp25 =  tmp0*tmp4*z[3]*z[4];
    tmp26 =  tmp0*tmp4*z[3]*z[5];
    tmp27 =  tmp19*tmp2*tmp3;
    tmp28 =  tmp0*tmp19*tmp4;
    tmp29 =  tmp2*tmp8*z[3]*z[5];
    tmp30 =  tmp2*tmp8*z[3]*z[4];
    tmp31 =  tmp10*tmp4*z[3]*z[5];
    tmp32 =  tmp10*tmp4*z[3]*z[4];
    tmp33 =  tmp10*tmp19*tmp4;
    tmp34 =  tmp19*tmp2*tmp8;
    out[0] =  weights[3];
    out[1] =  weights[0];
    out[2] =  weights[2];
    out[3] =  tmp0*weights[10] + tmp11*weights[7] + tmp2*weights[13] + tmp5*weights[6] + tmp6*weights[9] + tmp9*weights[11] + weights[5];
    out[4] =  tmp0*weights[8] + weights[1];
    out[5] =  tmp0*weights[12] + weights[4];
    out[6] =  tmp12*weights[38] + tmp13*weights[25] + tmp14*weights[39] + tmp15*weights[17] + tmp16*weights[23] + tmp17*weights[20] + tmp18*weights[29] + tmp20*weights[36] + tmp21*weights[24] + tmp22*weights[19] + tmp23*weights[15] + tmp24*weights[27] + tmp25*weights[30] + tmp26*weights[41] + tmp27*weights[34] + tmp28*weights[35] + tmp29*weights[16] + tmp30*weights[26] + tmp31*weights[32] + tmp32*weights[40] + tmp33*weights[14] + tmp34*weights[33] + weights[18]*z[7] + weights[21]*z[3] + weights[22]*z[4] + weights[28]*z[8] + weights[31]*z[5] + weights[37]*z[6] + z[9];
    out[7] =  weights[45];
    out[8] =  weights[42];
    out[9] =  weights[44];
    out[10] =  tmp0*weights[52] + tmp11*weights[49] + tmp2*weights[55] + tmp5*weights[48] + tmp6*weights[51] + tmp9*weights[53] + weights[47];
    out[11] =  tmp0*weights[50] + weights[43];
    out[12] =  tmp0*weights[54] + weights[46];
    out[13] =  tmp12*weights[80] + tmp13*weights[67] + tmp14*weights[81] + tmp15*weights[59] + tmp16*weights[65] + tmp17*weights[62] + tmp18*weights[71] + tmp20*weights[78] + tmp21*weights[66] + tmp22*weights[61] + tmp23*weights[57] + tmp24*weights[69] + tmp25*weights[72] + tmp26*weights[83] + tmp27*weights[76] + tmp28*weights[77] + tmp29*weights[58] + tmp30*weights[68] + tmp31*weights[74] + tmp32*weights[82] + tmp33*weights[56] + tmp34*weights[75] + weights[60]*z[7] + weights[63]*z[3] + weights[64]*z[4] + weights[70]*z[8] + weights[73]*z[5] + weights[79]*z[6] + z[10];
    out[14] =  weights[87];
    out[15] =  weights[84];
    out[16] =  weights[86];
    out[17] =  tmp0*weights[94] + tmp11*weights[91] + tmp2*weights[97] + tmp5*weights[90] + tmp6*weights[93] + tmp9*weights[95] + weights[89];
    out[18] =  tmp0*weights[92] + weights[85];
    out[19] =  tmp0*weights[96] + weights[88];
    out[20] =  tmp12*weights[122] + tmp13*weights[109] + tmp14*weights[123] + tmp15*weights[101] + tmp16*weights[107] + tmp17*weights[104] + tmp18*weights[113] + tmp20*weights[120] + tmp21*weights[108] + tmp22*weights[103] + tmp23*weights[99] + tmp24*weights[111] + tmp25*weights[114] + tmp26*weights[125] + tmp27*weights[118] + tmp28*weights[119] + tmp29*weights[100] + tmp30*weights[110] + tmp31*weights[116] + tmp32*weights[124] + tmp33*weights[98] + tmp34*weights[117] + weights[102]*z[7] + weights[105]*z[3] + weights[106]*z[4] + weights[112]*z[8] + weights[115]*z[5] + weights[121]*z[6] + z[11];
    out[21] =  weights[129];
    out[22] =  weights[126];
    out[23] =  weights[128];
    out[24] =  tmp0*weights[136] + tmp11*weights[133] + tmp2*weights[139] + tmp5*weights[132] + tmp6*weights[135] + tmp9*weights[137] + weights[131];
    out[25] =  tmp0*weights[134] + weights[127];
    out[26] =  tmp0*weights[138] + weights[130];
    out[27] =  tmp12*weights[164] + tmp13*weights[151] + tmp14*weights[165] + tmp15*weights[143] + tmp16*weights[149] + tmp17*weights[146] + tmp18*weights[155] + tmp20*weights[162] + tmp21*weights[150] + tmp22*weights[145] + tmp23*weights[141] + tmp24*weights[153] + tmp25*weights[156] + tmp26*weights[167] + tmp27*weights[160] + tmp28*weights[161] + tmp29*weights[142] + tmp30*weights[152] + tmp31*weights[158] + tmp32*weights[166] + tmp33*weights[140] + tmp34*weights[159] + weights[144]*z[7] + weights[147]*z[3] + weights[148]*z[4] + weights[154]*z[8] + weights[157]*z[5] + weights[163]*z[6] + z[12];
    out[28] =  weights[171];
    out[29] =  weights[168];
    out[30] =  weights[170];
    out[31] =  tmp0*weights[178] + tmp11*weights[175] + tmp2*weights[181] + tmp5*weights[174] + tmp6*weights[177] + tmp9*weights[179] + weights[173];
    out[32] =  tmp0*weights[176] + weights[169];
    out[33] =  tmp0*weights[180] + weights[172];
    out[34] =  tmp12*weights[206] + tmp13*weights[193] + tmp14*weights[207] + tmp15*weights[185] + tmp16*weights[191] + tmp17*weights[188] + tmp18*weights[197] + tmp20*weights[204] + tmp21*weights[192] + tmp22*weights[187] + tmp23*weights[183] + tmp24*weights[195] + tmp25*weights[198] + tmp26*weights[209] + tmp27*weights[202] + tmp28*weights[203] + tmp29*weights[184] + tmp30*weights[194] + tmp31*weights[200] + tmp32*weights[208] + tmp33*weights[182] + tmp34*weights[201] + weights[186]*z[7] + weights[189]*z[3] + weights[190]*z[4] + weights[196]*z[8] + weights[199]*z[5] + weights[205]*z[6] + z[13];
    out[35] =  weights[213];
    out[36] =  weights[210];
    out[37] =  weights[212];
    out[38] =  tmp0*weights[220] + tmp11*weights[217] + tmp2*weights[223] + tmp5*weights[216] + tmp6*weights[219] + tmp9*weights[221] + weights[215];
    out[39] =  tmp0*weights[218] + weights[211];
    out[40] =  tmp0*weights[222] + weights[214];
    out[41] =  tmp12*weights[248] + tmp13*weights[235] + tmp14*weights[249] + tmp15*weights[227] + tmp16*weights[233] + tmp17*weights[230] + tmp18*weights[239] + tmp20*weights[246] + tmp21*weights[234] + tmp22*weights[229] + tmp23*weights[225] + tmp24*weights[237] + tmp25*weights[240] + tmp26*weights[251] + tmp27*weights[244] + tmp28*weights[245] + tmp29*weights[226] + tmp30*weights[236] + tmp31*weights[242] + tmp32*weights[250] + tmp33*weights[224] + tmp34*weights[243] + weights[228]*z[7] + weights[231]*z[3] + weights[232]*z[4] + weights[238]*z[8] + weights[241]*z[5] + weights[247]*z[6] + z[14];
    
    
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

#endif /* ROBOTARM3DOF_DYNAMICS_H_ */

            