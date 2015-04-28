
#ifndef PENDULUM_DYNAMICS_BY_HAND_H_
#define PENDULUM_DYNAMICS_BY_HAND_H_

#include <iostream>
#include <math.h>
#define NX 14    // size of state space 
#define NU 7    // number of controls
#define NV 14    // number of virtual controls
#define NT 6    // number of constrained target state components

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/LU>
using namespace Eigen;        

float control_min[NU] = { -77.3, -160.6, -95.6, -29.4, -11.6, -11.6, -2.7 };
float control_max[NU] = {  77.3,  160.6,  95.6,  29.4,  11.6,  11.6,  2.7 };

#define EPS 1e-5

// Constraint values
float target_pos[3] = { -2.67059718e-02, -9.34824138e-05,  8.18595702e-01};
float target_vel[3] = { 3.15322, 0.0, 1.82051 }; // Want positive x and z velocity, 0 y velocity

float state_min[NX] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -150, -113, -157, -50, -275, -90, -172}; // In degrees
float state_max[NX] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  150,  113,  157, 180,   75,  90,  172};

namespace wam7dofarm {

    VectorXf rk45_DP(VectorXf (*f)(VectorXf, VectorXf, float[]), VectorXf x, VectorXf u, float delta, float weights[]) 
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
      VectorXf F1, F2, F3, F4, F5, F6, F7, y, z;

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
      //float epsilon = (y - z).norm();
      // std::cout << "Error: " << epsilon << ", Delta: " << delta << "\n";

      // Clip dynamics here: joint angles only
      // for(int i = NX/2; i < NX; ++i) {
      //   if (y(i) < state_min[i]*M_PI/180.0) {
      //     y(i) = state_min[i]*M_PI/180.0;
      //     y(i - NX/2) = 0; // Zero out joint velocity
      //     std::cout << "------------------------ CLIPPED DYNAMICS -------------------------------\n";
      //   }
      //   else if (y(i) > state_max[i]*M_PI/180.0) {
      //     y(i) = state_max[i]*M_PI/180.0;
      //     y(i - NX/2) = 0; // Zero out joint velocity
      //     std::cout << "------------------------ CLIPPED DYNAMICS -------------------------------\n";
      //   }
      // }

      return y;

    }

    VectorXf rk4(VectorXf (*f)(VectorXf, VectorXf, float[]), VectorXf x, VectorXf u, float delta, float weights[]) 
    {
        VectorXf k1 = delta*f(x, u, weights);
        VectorXf k2 = delta*f(x + .5*k1, u, weights);
        VectorXf k3 = delta*f(x + .5*k2, u, weights);
        VectorXf k4 = delta*f(x + k3, u, weights);

        VectorXf x_new = x + (k1 + 2*k2 + 2*k3 + k4)/6;
        return x_new;
    }

    VectorXf end_effector_pos(VectorXf x) {

      Vector3f pos;
      VectorXf orientation(9);
      pos.setZero(); orientation.setZero();

      VectorXf q(7);
      q = x.tail(7); // x = [dq, q]

      float s1 = sin(q(0));
      float s2 = sin(q(1));
      float s3 = sin(q(2));
      float s4 = sin(q(3));
      float s5 = sin(q(4));
      float s6 = sin(q(5));
      //float s7 = sin(q(6));

      float c1 = cos(q(0));
      float c2 = cos(q(1));
      float c3 = cos(q(2));
      float c4 = cos(q(3));
      float c5 = cos(q(4));
      float c6 = cos(q(5));
      //float c7 = cos(q(6));

      // This code was printed from sympybotics. The joints angles are q1 to q7.

      // Position vector
      pos << 0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 - 0.045*(-s1*s3 + c1*c2*c3)*c4 - 0.045*s1*s3 + 0.045*s2*s4*c1 + 0.3*s2*c1*c4 + 0.55*s2*c1 + 0.045*c1*c2*c3, 
      0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 - 0.045*(s1*c2*c3 + s3*c1)*c4 + 0.045*s1*s2*s4 + 0.3*s1*s2*c4 + 0.55*s1*s2 + 0.045*s1*c2*c3 + 0.045*s3*c1,
      0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.045*s2*c3*c4 - 0.045*s2*c3 + 0.045*s4*c2 + 0.3*c2*c4 + 0.55*c2;



      // Rotation matrix, in row major format
      // orientation << ((((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*c6 + (-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*s6)*c7 + (-((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*s5 + (-s1*c3 - s3*c1*c2)*c5)*s7,
      // -((((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*c6 + (-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*s6)*s7 + (-((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*s5 + (-s1*c3 - s3*c1*c2)*c5)*c7,
      // (((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - (-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6,
      // ((((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*c6 + (-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*s6)*c7 + (-((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*s5 + (-s1*s3*c2 + c1*c3)*c5)*s7,
      // -((((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*c6 + (-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*s6)*s7 + (-((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*s5 + (-s1*s3*c2 + c1*c3)*c5)*c7,
      // (((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - (-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6,
      // (((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*c6 + (s2*s4*c3 - c2*c4)*s6)*c7 + (-(-s2*c3*c4 - s4*c2)*s5 + s2*s3*c5)*s7,
      // -(((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*c6 + (s2*s4*c3 - c2*c4)*s6)*s7 + (-(-s2*c3*c4 - s4*c2)*s5 + s2*s3*c5)*c7,
      // ((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - (s2*s4*c3 - c2*c4)*c6;

      return pos;

    }

    VectorXf end_effector_lin_vel(VectorXf x) {

      Vector3f lin_vel;
      Vector3f ang_vel;
      VectorXf all_vels(6); // [lin_vel, ang_vel]
      lin_vel.setZero(); ang_vel.setZero(); all_vels.setZero();

      VectorXf q(7);
      q = x.tail(7); // x = [dq, q]
      VectorXf dq(7);
      dq = x.head(7); // x = [dq, q]

      Matrix<float, 6, 7> J_q;
      J_q.setZero();

      float s1 = sin(q(0));
      float s2 = sin(q(1));
      float s3 = sin(q(2));
      float s4 = sin(q(3));
      float s5 = sin(q(4));
      float s6 = sin(q(5));
      //float s7 = sin(q(6));

      float c1 = cos(q(0));
      float c2 = cos(q(1));
      float c3 = cos(q(2));
      float c4 = cos(q(3));
      float c5 = cos(q(4));
      float c6 = cos(q(5));
      //float c7 = cos(q(6));

      // This code was printed from sympybotics. The joint angles are q1 to q7.

      // Kinematic jacobian of the end effector
      J_q << -0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 + 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 - 0.3*(s1*c2*c3 + s3*c1)*s4 + 0.045*(s1*c2*c3 + s3*c1)*c4 - 0.045*s1*s2*s4 - 0.3*s1*s2*c4 - 0.55*s1*s2 - 0.045*s1*c2*c3 - 0.045*s3*c1,
      (0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.045*s2*c3*c4 - 0.045*s2*c3 + 0.045*s4*c2 + 0.3*c2*c4 + 0.55*c2)*c1,
      (0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.045*s2*c3*c4 - 0.045*s2*c3 + 0.045*s4*c2 + 0.3*c2*c4 + 0.55*c2)*s1*s2 - (0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 - 0.045*(s1*c2*c3 + s3*c1)*c4 + 0.045*s1*s2*s4 + 0.3*s1*s2*c4 + 0.55*s1*s2 + 0.045*s1*c2*c3 + 0.045*s3*c1)*c2,
      (-s1*s3*c2 + c1*c3)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.045*s2*c3*c4 + 0.045*s4*c2 + 0.3*c2*c4) - (0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 - 0.045*(s1*c2*c3 + s3*c1)*c4 + 0.045*s1*s2*s4 + 0.3*s1*s2*c4)*s2*s3,
      ((s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.3*c2*c4) - (-s2*s4*c3 + c2*c4)*(0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 + 0.3*s1*s2*c4),
      -(0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6)*(-(-s2*c3*c4 - s4*c2)*s5 + s2*s3*c5) + (-((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*s5 + (-s1*s3*c2 + c1*c3)*c5)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6),
      -(0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6)*(((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - (s2*s4*c3 - c2*c4)*c6) + ((((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - (-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6),
      0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 - 0.045*(-s1*s3 + c1*c2*c3)*c4 - 0.045*s1*s3 + 0.045*s2*s4*c1 + 0.3*s2*c1*c4 + 0.55*s2*c1 + 0.045*c1*c2*c3,
      (0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.045*s2*c3*c4 - 0.045*s2*c3 + 0.045*s4*c2 + 0.3*c2*c4 + 0.55*c2)*s1,
      -(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.045*s2*c3*c4 - 0.045*s2*c3 + 0.045*s4*c2 + 0.3*c2*c4 + 0.55*c2)*s2*c1 + (0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 - 0.045*(-s1*s3 + c1*c2*c3)*c4 - 0.045*s1*s3 + 0.045*s2*s4*c1 + 0.3*s2*c1*c4 + 0.55*s2*c1 + 0.045*c1*c2*c3)*c2,
      -(-s1*c3 - s3*c1*c2)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.045*s2*c3*c4 + 0.045*s4*c2 + 0.3*c2*c4) + (0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 - 0.045*(-s1*s3 + c1*c2*c3)*c4 + 0.045*s2*s4*c1 + 0.3*s2*c1*c4)*s2*s3,
      -((-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.3*c2*c4) + (-s2*s4*c3 + c2*c4)*(0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 + 0.3*s2*c1*c4),
      (0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6)*(-(-s2*c3*c4 - s4*c2)*s5 + s2*s3*c5) - (-((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*s5 + (-s1*c3 - s3*c1*c2)*c5)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6),
      (0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6)*(((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - (s2*s4*c3 - c2*c4)*c6) - ((((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - (-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6),
      0,
      -(0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 - 0.045*(-s1*s3 + c1*c2*c3)*c4 - 0.045*s1*s3 + 0.045*s2*s4*c1 + 0.3*s2*c1*c4 + 0.55*s2*c1 + 0.045*c1*c2*c3)*c1 - (0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 - 0.045*(s1*c2*c3 + s3*c1)*c4 + 0.045*s1*s2*s4 + 0.3*s1*s2*c4 + 0.55*s1*s2 + 0.045*s1*c2*c3 + 0.045*s3*c1)*s1,
      -(0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 - 0.045*(-s1*s3 + c1*c2*c3)*c4 - 0.045*s1*s3 + 0.045*s2*s4*c1 + 0.3*s2*c1*c4 + 0.55*s2*c1 + 0.045*c1*c2*c3)*s1*s2 + (0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 - 0.045*(s1*c2*c3 + s3*c1)*c4 + 0.045*s1*s2*s4 + 0.3*s1*s2*c4 + 0.55*s1*s2 + 0.045*s1*c2*c3 + 0.045*s3*c1)*s2*c1,
      (-s1*c3 - s3*c1*c2)*(0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 - 0.045*(s1*c2*c3 + s3*c1)*c4 + 0.045*s1*s2*s4 + 0.3*s1*s2*c4) - (-s1*s3*c2 + c1*c3)*(0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 - 0.045*(-s1*s3 + c1*c2*c3)*c4 + 0.045*s2*s4*c1 + 0.3*s2*c1*c4),
      ((-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*(0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 + 0.3*s1*s2*c4) - ((s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*(0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 + 0.3*s2*c1*c4),
      -(0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6)*(-((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*s5 + (-s1*s3*c2 + c1*c3)*c5) + (0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6)*(-((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*s5 + (-s1*c3 - s3*c1*c2)*c5),
      -(0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6)*((((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - (-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6) + ((((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - (-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6)*(0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6),
      0,
      -s1,
      s2*c1,
      -s1*c3 - s3*c1*c2,
      (-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4,
      -((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*s5 + (-s1*c3 - s3*c1*c2)*c5,
      (((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - (-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6,
      0,
      c1,
      s1*s2,
      -s1*s3*c2 + c1*c3,
      (s1*c2*c3 + s3*c1)*s4 + s1*s2*c4,
      -((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*s5 + (-s1*s3*c2 + c1*c3)*c5,
      (((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - (-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6,
      1,
      0,
      c2,
      s2*s3,
      -s2*s4*c3 + c2*c4,
      -(-s2*c3*c4 - s4*c2)*s5 + s2*s3*c5,
      ((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - (s2*s4*c3 - c2*c4)*c6;

      all_vels = J_q * dq;
      lin_vel = all_vels.head(3);
      ang_vel = all_vels.tail(3);

      return lin_vel;
    }

    void M( float* M_out, const float* parms, const float* q )
    {
      float x0 = cos(q[1]);
      float x1 = sin(q[2]);
      float x2 = cos(q[2]);
      float x3 = 0.045*((x1)*(x1)) + 0.045*((x2)*(x2));
      float x4 = cos(q[3]);
      float x5 = -x4;
      float x6 = -x0;
      float x7 = x5*x6;
      float x8 = ((x4)*(x4));
      float x9 = sin(q[3]);
      float x10 = ((x9)*(x9));
      float x11 = -0.045*x10 - 0.045*x8;
      float x12 = sin(q[1]);
      float x13 = -x12;
      float x14 = x13*x2;
      float x15 = x0*x3 - 0.55*x14;
      float x16 = x14*x9;
      float x17 = x11*x16 + x11*x7 + x15;
      float x18 = -parms[46];
      float x19 = x16 + x7;
      float x20 = -x19;
      float x21 = sin(q[4]);
      float x22 = x1*x13;
      float x23 = -x22;
      float x24 = x21*x23;
      float x25 = x14*x4 + x6*x9;
      float x26 = cos(q[4]);
      float x27 = x25*x26;
      float x28 = -x21;
      float x29 = -0.55*x22;
      float x30 = x23*x3;
      float x31 = x29*x4 + x30*x9;
      float x32 = x17*x26 - 0.3*x24 - 0.3*x27 + x28*x31;
      float x33 = x24 + x27;
      float x34 = cos(q[5]);
      float x35 = -x34;
      float x36 = sin(q[5]);
      float x37 = x20*x35 + x33*x36;
      float x38 = cos(q[6]);
      float x39 = sin(q[6]);
      float x40 = -x39;
      float x41 = 0.045*x10 + 0.045*x8;
      float x42 = x23*x41 + x29*x9 + x30*x5;
      float x43 = -x42;
      float x44 = x23*x26;
      float x45 = x21*x25;
      float x46 = x17*x21 + x26*x31 + 0.3*x44 - 0.3*x45;
      float x47 = x34*x46 + x36*x43;
      float x48 = x44 - x45;
      float x49 = x39*x48;
      float x50 = x20*x36 + x33*x34;
      float x51 = x38*x50;
      float x52 = x32*x38 + x40*x47 - 0.06*x49 - 0.06*x51;
      float x53 = -parms[68];
      float x54 = x49 + x51;
      float x55 = parms[66]*x37 + parms[69]*x52 + x53*x54;
      float x56 = x38*x55;
      float x57 = -parms[58];
      float x58 = -parms[67];
      float x59 = x39*x50;
      float x60 = x38*x48;
      float x61 = -x59 + x60;
      float x62 = x32*x39 + x38*x47 - 0.06*x59 + 0.06*x60;
      float x63 = parms[68]*x61 + parms[69]*x62 + x37*x58;
      float x64 = parms[47]*x33 + parms[49]*x32 + parms[56]*x37 + parms[59]*x32 + x18*x20 + x39*x63 + x50*x57 + x56;
      float x65 = x26*x64;
      float x66 = -parms[57];
      float x67 = x38*x63;
      float x68 = parms[58]*x48 + parms[59]*x47 + x37*x66 + x40*x55 + x67;
      float x69 = -parms[47];
      float x70 = -parms[56];
      float x71 = x35*x43 + x36*x46;
      float x72 = -parms[66];
      float x73 = parms[57]*x50 + parms[59]*x71 + parms[67]*x54 + parms[69]*x71 + x48*x70 + x61*x72;
      float x74 = parms[48]*x20 + parms[49]*x46 + x34*x68 + x36*x73 + x48*x69;
      float x75 = -parms[38];
      float x76 = parms[36]*x19 + parms[39]*x17 + x21*x74 + x25*x75 + x65;
      float x77 = -parms[26];
      float x78 = parms[27]*x14 + parms[29]*x15 + x6*x77 + x76;
      float x79 = x11*x76;
      float x80 = -0.06*x39;
      float x81 = parms[60]*x54 + parms[61]*x61 + parms[62]*x37 + parms[67]*x71 + x52*x53;
      float x82 = parms[61]*x54 + parms[63]*x61 + parms[64]*x37 + parms[68]*x62 + x71*x72;
      float x83 = parms[50]*x50 + parms[51]*x48 + parms[52]*x37 + parms[57]*x71 + x32*x57 + x38*x81 + x40*x82 - 0.06*x56 + x63*x80;
      float x84 = parms[62]*x54 + parms[64]*x61 + parms[65]*x37 + parms[66]*x52 + x58*x62;
      float x85 = parms[52]*x50 + parms[54]*x48 + parms[55]*x37 + parms[56]*x32 + x47*x66 + x84;
      float x86 = -parms[48];
      float x87 = parms[40]*x33 + parms[41]*x20 + parms[42]*x48 + parms[47]*x32 + x34*x83 + x36*x85 + x43*x86;
      float x88 = -0.3*x21;
      float x89 = parms[51]*x50 + parms[53]*x48 + parms[54]*x37 + parms[58]*x47 + x38*x82 + x39*x81 + x55*x80 + 0.06*x67 + x70*x71;
      float x90 = parms[42]*x33 + parms[44]*x20 + parms[45]*x48 + parms[46]*x43 + x46*x69 + x89;
      float x91 = parms[30]*x25 + parms[31]*x23 + parms[32]*x19 + parms[37]*x42 + x17*x75 + x26*x87 + x28*x90 - 0.3*x65 + x74*x88;
      float x92 = -parms[41]*x33 - parms[43]*x20 - parms[44]*x48 - parms[48]*x46 - x18*x32 - x35*x85 - x36*x83;
      float x93 = -parms[37];
      float x94 = parms[32]*x25 + parms[34]*x23 + parms[35]*x19 + parms[36]*x17 + x31*x93 + x92;
      float x95 = -parms[21]*x14 - parms[23]*x6 - parms[24]*x23 - parms[28]*x29 - x15*x77 - x5*x79 - x5*x94 - x9*x91;
      float x96 = -x23;
      float x97 = x26*x74;
      float x98 = parms[37]*x20 + parms[38]*x23 + parms[39]*x31 + x28*x64 + x97;
      float x99 = parms[36]*x96 + parms[37]*x25 + parms[39]*x42 - parms[46]*x48 - parms[49]*x43 - x33*x86 - x35*x73 - x36*x68;
      float x100 = parms[27]*x96 + parms[28]*x6 + parms[29]*x29 + x4*x98 + x9*x99;
      float x101 = -0.55*x1;
      float x102 = -parms[28];
      float x103 = parms[20]*x14 + parms[21]*x6 + parms[22]*x23 + parms[27]*x15 + x102*x30 + x4*x91 + x79*x9 + x9*x94;
      float x104 = -x1;
      float x105 = x3*(parms[26]*x23 + parms[29]*x30 + x102*x14 + x5*x99 + x9*x98);
      float x106 = parms[31]*x25 + parms[33]*x23 + parms[34]*x19 + parms[36]*x43 + parms[38]*x31 + x21*x87 + x26*x90 + x64*x88 + 0.3*x97;
      float x107 = -parms[27];
      float x108 = parms[22]*x14 + parms[24]*x6 + parms[25]*x23 + parms[26]*x30 + x106 + x107*x29 + x41*x99;
      float x109 = parms[11]*x13 + parms[14]*x0 + x1*x103 + 0.55*x100*x2 + x101*x78 + x105*x2 + x108*x2;
      float x110 = 0.045*x78 + x95;
      float x111 = x106 + 0.045*x99;
      float x112 = x1*x4;
      float x113 = x2*x26;
      float x114 = x112*x28 + x113;
      float x115 = x112*x26;
      float x116 = x115 + x2*x21;
      float x117 = x1*x9;
      float x118 = -x117;
      float x119 = x116*x34 + x118*x36;
      float x120 = x101 + x11*x117;
      float x121 = 0.55*x2;
      float x122 = x2*x3;
      float x123 = x121*x4 + x122*x9;
      float x124 = x112*x88 + 0.3*x113 + x120*x21 + x123*x26;
      float x125 = x121*x9 + x122*x5 + x2*x41;
      float x126 = -x125;
      float x127 = x124*x36 + x126*x35;
      float x128 = x119*x38;
      float x129 = x114*x39 + x128;
      float x130 = x114*x38;
      float x131 = x119*x40 + x130;
      float x132 = parms[57]*x119 + parms[59]*x127 + parms[67]*x129 + parms[69]*x127 + x114*x70 + x131*x72;
      float x133 = x116*x36 + x118*x35;
      float x134 = x124*x34 + x126*x36;
      float x135 = -0.3*x115 + x120*x26 + x123*x28 + x2*x88;
      float x136 = x119*x80 + 0.06*x130 + x134*x38 + x135*x39;
      float x137 = parms[68]*x131 + parms[69]*x136 + x133*x58;
      float x138 = x137*x38;
      float x139 = x114*x80 - 0.06*x128 + x134*x40 + x135*x38;
      float x140 = parms[66]*x133 + parms[69]*x139 + x129*x53;
      float x141 = parms[58]*x114 + parms[59]*x134 + x133*x66 + x138 + x140*x40;
      float x142 = parms[48]*x118 + parms[49]*x124 + x114*x69 + x132*x36 + x141*x34;
      float x143 = x140*x38;
      float x144 = parms[47]*x116 + parms[49]*x135 + parms[56]*x133 + parms[59]*x135 + x118*x18 + x119*x57 + x137*x39 + x143;
      float x145 = x144*x26;
      float x146 = parms[36]*x117 + parms[39]*x120 + x112*x75 + x142*x21 + x145;
      float x147 = parms[27]*x1 + parms[29]*x101 + x146;
      float x148 = parms[60]*x129 + parms[61]*x131 + parms[62]*x133 + parms[67]*x127 + x139*x53;
      float x149 = parms[61]*x129 + parms[63]*x131 + parms[64]*x133 + parms[68]*x136 + x127*x72;
      float x150 = parms[51]*x119 + parms[53]*x114 + parms[54]*x133 + parms[58]*x134 + x127*x70 + 0.06*x138 + x140*x80 + x148*x39 + x149*x38;
      float x151 = parms[42]*x116 + parms[44]*x118 + parms[45]*x114 + parms[46]*x126 + x124*x69 + x150;
      float x152 = parms[50]*x119 + parms[51]*x114 + parms[52]*x133 + parms[57]*x127 + x135*x57 + x137*x80 - 0.06*x143 + x148*x38 + x149*x40;
      float x153 = parms[62]*x129 + parms[64]*x131 + parms[65]*x133 + parms[66]*x139 + x136*x58;
      float x154 = parms[52]*x119 + parms[54]*x114 + parms[55]*x133 + parms[56]*x135 + x134*x66 + x153;
      float x155 = parms[40]*x116 + parms[41]*x118 + parms[42]*x114 + parms[47]*x135 + x126*x86 + x152*x34 + x154*x36;
      float x156 = parms[30]*x112 + parms[31]*x2 + parms[32]*x117 + parms[37]*x125 + x120*x75 + x142*x88 - 0.3*x145 + x151*x28 + x155*x26;
      float x157 = x11*x146;
      float x158 = -parms[41]*x116 - parms[43]*x118 - parms[44]*x114 - parms[48]*x124 - x135*x18 - x152*x36 - x154*x35;
      float x159 = parms[32]*x112 + parms[34]*x2 + parms[35]*x117 + parms[36]*x120 + x123*x93 + x158;
      float x160 = -parms[36];
      float x161 = parms[37]*x112 + parms[39]*x125 - parms[46]*x114 - parms[49]*x126 - x116*x86 - x132*x35 - x141*x36 + x160*x2;
      float x162 = x142*x26;
      float x163 = parms[31]*x112 + parms[33]*x2 + parms[34]*x117 + parms[36]*x126 + parms[38]*x123 + x144*x88 + x151*x26 + x155*x21 + 0.3*x162;
      float x164 = parms[37]*x118 + parms[38]*x2 + parms[39]*x123 + x144*x28 + x162;
      float x165 = -parms[21]*x1 - parms[24]*x2 - parms[28]*x121 - x101*x77 + 0.045*x147 - x156*x9 - x157*x5 - x159*x5;
      float x166 = 0.045*x161 + x163;
      float x167 = x11*x4 + 0.045;
      float x168 = -x9;
      float x169 = x168*x28;
      float x170 = x168*x26;
      float x171 = x167*x26 - 0.3*x170;
      float x172 = x167*x21 + x168*x88;
      float x173 = x172*x34;
      float x174 = x170*x34 + x36*x5;
      float x175 = x169*x38;
      float x176 = x171*x39 + x173*x38 + x174*x80 + 0.06*x175;
      float x177 = x170*x36 + x35*x5;
      float x178 = x174*x40 + x175;
      float x179 = parms[68]*x178 + parms[69]*x176 + x177*x58;
      float x180 = x179*x38;
      float x181 = x174*x38;
      float x182 = x169*x39 + x181;
      float x183 = x169*x80 + x171*x38 + x173*x40 - 0.06*x181;
      float x184 = parms[66]*x177 + parms[69]*x183 + x182*x53;
      float x185 = parms[58]*x169 + parms[59]*x173 + x177*x66 + x180 + x184*x40;
      float x186 = x172*x36;
      float x187 = parms[57]*x174 + parms[59]*x186 + parms[67]*x182 + parms[69]*x186 + x169*x70 + x178*x72;
      float x188 = parms[48]*x5 + parms[49]*x172 + x169*x69 + x185*x34 + x187*x36;
      float x189 = x184*x38;
      float x190 = parms[47]*x170 + parms[49]*x171 + parms[56]*x177 + parms[59]*x171 + x174*x57 + x179*x39 + x18*x5 + x189;
      float x191 = x190*x26;
      float x192 = parms[36]*x4 + parms[39]*x167 + x168*x75 + x188*x21 + x191;
      float x193 = parms[62]*x182 + parms[64]*x178 + parms[65]*x177 + parms[66]*x183 + x176*x58;
      float x194 = parms[52]*x174 + parms[54]*x169 + parms[55]*x177 + parms[56]*x171 + x173*x66 + x193;
      float x195 = parms[61]*x182 + parms[63]*x178 + parms[64]*x177 + parms[68]*x176 + x186*x72;
      float x196 = parms[60]*x182 + parms[61]*x178 + parms[62]*x177 + parms[67]*x186 + x183*x53;
      float x197 = parms[50]*x174 + parms[51]*x169 + parms[52]*x177 + parms[57]*x186 + x171*x57 + x179*x80 - 0.06*x189 + x195*x40 + x196*x38;
      float x198 = parms[40]*x170 + parms[41]*x5 + parms[42]*x169 + parms[47]*x171 + x194*x36 + x197*x34;
      float x199 = parms[51]*x174 + parms[53]*x169 + parms[54]*x177 + parms[58]*x173 + 0.06*x180 + x184*x80 + x186*x70 + x195*x38 + x196*x39;
      float x200 = parms[42]*x170 + parms[44]*x5 + parms[45]*x169 + x172*x69 + x199;
      float x201 = -parms[41]*x170 - parms[43]*x5 - parms[44]*x169 - parms[48]*x172 - x171*x18 - x194*x35 - x197*x36;
      float x202 = parms[31]*x168 + parms[34]*x4 + 0.045*parms[37]*x168 - 0.045*parms[46]*x169 - 0.045*x170*x86 - 0.045*x185*x36 - 0.045*x187*x35 + 0.3*x188*x26 + x190*x88 + x198*x21 + x200*x26;
      float x203 = x21*x34;
      float x204 = x21*x36;
      float x205 = x26*x38;
      float x206 = x203*x40 + x205;
      float x207 = 0.3*x26;
      float x208 = x207*x34 - 0.045*x36;
      float x209 = x203*x80 + 0.06*x205 + x208*x38 + x39*x88;
      float x210 = parms[68]*x206 + parms[69]*x209 + x204*x58;
      float x211 = x203*x38;
      float x212 = x208*x40 - 0.06*x211 + x26*x80 + x38*x88;
      float x213 = x211 + x26*x39;
      float x214 = parms[66]*x204 + parms[69]*x212 + x213*x53;
      float x215 = x214*x38;
      float x216 = x210*x38;
      float x217 = parms[58]*x26 + parms[59]*x208 + x204*x66 + x214*x40 + x216;
      float x218 = x207*x36 + 0.045*x34;
      float x219 = parms[57]*x203 + parms[59]*x218 + parms[67]*x213 + parms[69]*x218 + x206*x72 + x26*x70;
      float x220 = parms[61]*x213 + parms[63]*x206 + parms[64]*x204 + parms[68]*x209 + x218*x72;
      float x221 = parms[60]*x213 + parms[61]*x206 + parms[62]*x204 + parms[67]*x218 + x212*x53;
      float x222 = parms[50]*x203 + parms[51]*x26 + parms[52]*x204 + parms[57]*x218 + x210*x80 - 0.06*x215 + x220*x40 + x221*x38 + x57*x88;
      float x223 = parms[62]*x213 + parms[64]*x206 + parms[65]*x204 + parms[66]*x212 + x209*x58;
      float x224 = parms[52]*x203 + parms[54]*x26 + parms[55]*x204 + parms[56]*x88 + x208*x66 + x223;
      float x225 = parms[51]*x203 + parms[53]*x26 + parms[54]*x204 + parms[58]*x208 + x214*x80 + 0.06*x216 + x218*x70 + x220*x38 + x221*x39;
      float x226 = -parms[41]*x21 - parms[44]*x26 - parms[48]*x207 - x18*x88 - x222*x36 - x224*x35;
      float x227 = -x36;
      float x228 = x227*x80;
      float x229 = x227*x40;
      float x230 = parms[67]*x35 + parms[68]*x229 + parms[69]*x228;
      float x231 = x227*x38;
      float x232 = -0.06*x231;
      float x233 = parms[60]*x231 + parms[61]*x229 + parms[62]*x34 + x232*x53;
      float x234 = parms[61]*x231 + parms[63]*x229 + parms[64]*x34 + parms[68]*x228;
      float x235 = parms[66]*x34 + parms[69]*x232 + x231*x53;
      float x236 = parms[62]*x231 + parms[64]*x229 + parms[65]*x34 + parms[66]*x232 + x228*x58;
      float x237 = parms[51]*x227 + parms[54]*x34 + 0.06*x230*x38 + x233*x39 + x234*x38 + x235*x80;
      float x238 = 0.06*x38;
      float x239 = parms[62]*x39 + parms[64]*x38 + parms[66]*x80 + x238*x58;
    //
      M_out[0] = parms[3] - x12*(parms[10]*x13 + parms[12]*x0 + x100*x101 + x103*x2 + x104*x105 + x104*x108 - 0.55*x2*x78) - x6*(parms[12]*x13 + parms[15]*x0 + x3*x78 + x95);
      M_out[1] = x109;
      M_out[2] = x110;
      M_out[3] = x111;
      M_out[4] = x92;
      M_out[5] = x89;
      M_out[6] = x84;
      M_out[7] = x109;
      M_out[8] = parms[13] + x1*(parms[20]*x1 + parms[22]*x2 + parms[27]*x101 + x102*x122 + x156*x4 + x157*x9 + x159*x9) + x101*x147 + x121*(parms[29]*x121 + x107*x2 + x161*x9 + x164*x4) + x122*(parms[26]*x2 + parms[29]*x122 + x1*x102 + x161*x5 + x164*x9) + x2*(parms[22]*x1 + parms[25]*x2 + parms[26]*x122 + x107*x121 + x161*x41 + x163);
      M_out[9] = x165;
      M_out[10] = x166;
      M_out[11] = x158;
      M_out[12] = x150;
      M_out[13] = x153;
      M_out[14] = x110;
      M_out[15] = x165;
      M_out[16] = parms[23] + 0.09*parms[26] + 0.002025*parms[29] - x11*x192*x5 + 0.045*x192 - x5*(parms[32]*x168 + parms[35]*x4 + parms[36]*x167 + x201) - x9*(parms[30]*x168 + parms[32]*x4 + x167*x75 + x188*x88 - 0.3*x191 + x198*x26 + x200*x28);
      M_out[17] = x202;
      M_out[18] = x201;
      M_out[19] = x199;
      M_out[20] = x193;
      M_out[21] = x111;
      M_out[22] = x166;
      M_out[23] = x202;
      M_out[24] = parms[33] - 0.045*parms[36] + 0.002025*parms[39] - 0.045*parms[46]*x26 + 0.002025*parms[49] + 0.045*x160 + x207*(parms[49]*x207 + x217*x34 + x219*x36 + x26*x69) - 0.045*x21*x86 + x21*(parms[40]*x21 + parms[42]*x26 + parms[47]*x88 + 0.045*parms[48] + x222*x34 + x224*x36) - 0.045*x217*x36 - 0.045*x219*x35 + x26*(parms[42]*x21 + parms[45]*x26 - 0.045*parms[46] + x207*x69 + x225) + x88*(parms[47]*x21 + parms[49]*x88 + parms[56]*x204 + parms[59]*x88 + x203*x57 + x210*x39 + x215);
      M_out[25] = x226;
      M_out[26] = x225;
      M_out[27] = x223;
      M_out[28] = x92;
      M_out[29] = x158;
      M_out[30] = x201;
      M_out[31] = x226;
      M_out[32] = parms[43] - x35*(parms[52]*x227 + parms[55]*x34 + x236) - x36*(parms[50]*x227 + parms[52]*x34 + x230*x80 + x233*x38 + x234*x40 - 0.06*x235*x38);
      M_out[33] = x237;
      M_out[34] = x236;
      M_out[35] = x89;
      M_out[36] = x150;
      M_out[37] = x199;
      M_out[38] = x225;
      M_out[39] = x237;
      M_out[40] = parms[53] + x238*(parms[68]*x38 + parms[69]*x238) + x38*(parms[61]*x39 + parms[63]*x38 + parms[68]*x238) + x39*(parms[60]*x39 + parms[61]*x38 + x53*x80) + x80*(parms[69]*x80 + x39*x53);
      M_out[41] = x239;
      M_out[42] = x84;
      M_out[43] = x153;
      M_out[44] = x193;
      M_out[45] = x223;
      M_out[46] = x236;
      M_out[47] = x239;
      M_out[48] = parms[65];
    //
      return;
    }

    void c( float* c_out, const float* parms, const float* q, const float* dq )
    {
      float x0 = -dq[0];
      float x1 = -cos(q[1]);
      float x2 = x0*x1;
      float x3 = -x2;
      float x4 = dq[1]*x3;
      float x5 = sin(q[1]);
      float x6 = x0*x5;
      float x7 = dq[1]*parms[11] + parms[10]*x6 + parms[12]*x2;
      float x8 = dq[1]*x6;
      float x9 = cos(q[2]);
      float x10 = sin(q[2]);
      float x11 = 0.045*((x10)*(x10)) + 0.045*((x9)*(x9));
      float x12 = x10*x6;
      float x13 = dq[1]*x9;
      float x14 = -0.55*x12 + 0.55*x13;
      float x15 = -dq[2];
      float x16 = x15 + x3;
      float x17 = -x12;
      float x18 = x13 + x17;
      float x19 = -parms[27];
      float x20 = parms[28]*x16 + parms[29]*x14 + x18*x19;
      float x21 = -x16;
      float x22 = sin(q[3]);
      float x23 = dq[1]*x10;
      float x24 = x6*x9;
      float x25 = x23 + x24;
      float x26 = cos(q[3]);
      float x27 = x16*x22 + x25*x26;
      float x28 = -x27;
      float x29 = 0.045*dq[3];
      float x30 = x11*x13 + x11*x17;
      float x31 = -x26;
      float x32 = ((x22)*(x22));
      float x33 = ((x26)*(x26));
      float x34 = 0.045*x32 + 0.045*x33;
      float x35 = x14*x22 + x18*x34 + x30*x31;
      float x36 = x29 + x35;
      float x37 = dq[3] + x18;
      float x38 = -x37;
      float x39 = parms[36]*x38 + parms[37]*x27 + parms[39]*x36;
      float x40 = cos(q[4]);
      float x41 = -x15;
      float x42 = x4*x9;
      float x43 = x18*x41 + x42;
      float x44 = x22*x43;
      float x45 = -x8;
      float x46 = x31*x45;
      float x47 = dq[3]*x27 + x44 + x46;
      float x48 = -x47;
      float x49 = -parms[46];
      float x50 = sin(q[4]);
      float x51 = x37*x40;
      float x52 = x28*x50 + x51;
      float x53 = -dq[4];
      float x54 = -x53;
      float x55 = -dq[3];
      float x56 = x22*x25;
      float x57 = x16*x31;
      float x58 = x56 + x57;
      float x59 = x22*x45 + x26*x43 + x55*x58;
      float x60 = x40*x59;
      float x61 = x10*x4;
      float x62 = -x61;
      float x63 = x15*x25 + x62;
      float x64 = x50*x63;
      float x65 = x52*x54 + x60 + x64;
      float x66 = -parms[47];
      float x67 = -x58;
      float x68 = x53 + x67;
      float x69 = x14*x26 + x22*x30;
      float x70 = 0.045*dq[2];
      float x71 = x11*x2 - 0.55*x23 - 0.55*x24;
      float x72 = x70 + x71;
      float x73 = -0.045*x32 - 0.045*x33;
      float x74 = x56*x73 + x57*x73 + x72;
      float x75 = -0.3*x50;
      float x76 = x27*x75 + x40*x69 + x50*x74 + 0.3*x51;
      float x77 = parms[48]*x68 + parms[49]*x76 + x52*x66;
      float x78 = -x68;
      float x79 = x11*x8 + x14*x15 - 0.55*x42;
      float x80 = x28*x29 + x44*x73 + x46*x73 + x79;
      float x81 = x3*x70 + x41*x71 - 0.55*x61;
      float x82 = -x25;
      float x83 = x11*x62 + x70*x82;
      float x84 = x18*x29 + x22*x83 + x26*x81 + x35*x55;
      float x85 = -x50;
      float x86 = x40*x80 + x53*x76 - 0.3*x60 - 0.3*x64 + x84*x85;
      float x87 = x27*x40;
      float x88 = x37*x50;
      float x89 = x87 + x88;
      float x90 = -x36;
      float x91 = -parms[48];
      float x92 = parms[46]*x52 + parms[49]*x90 + x89*x91;
      float x93 = cos(q[5]);
      float x94 = sin(q[5]);
      float x95 = -x93;
      float x96 = x68*x95 + x89*x94;
      float x97 = -dq[5];
      float x98 = x48*x94 + x65*x93 + x96*x97;
      float x99 = -parms[58];
      float x100 = cos(q[6]);
      float x101 = dq[5] + x52;
      float x102 = sin(q[6]);
      float x103 = x101*x102;
      float x104 = x68*x94 + x89*x93;
      float x105 = x100*x104;
      float x106 = x103 + x105;
      float x107 = x100*x101;
      float x108 = x102*x104;
      float x109 = x107 - x108;
      float x110 = -parms[66];
      float x111 = x76*x94 + x90*x95;
      float x112 = parms[67]*x106 + parms[69]*x111 + x109*x110;
      float x113 = -x106;
      float x114 = dq[5]*x104 + x48*x95 + x65*x94;
      float x115 = x40*x74 + x69*x85 - 0.3*x87 - 0.3*x88;
      float x116 = x76*x93 + x90*x94;
      float x117 = x100*x116 + x102*x115 + 0.06*x107 - 0.06*x108;
      float x118 = -dq[6];
      float x119 = x40*x63;
      float x120 = x115*x54 + 0.3*x119 + x40*x84 + x50*x80 + x59*x75;
      float x121 = dq[3]*x69 + x22*x81 + x31*x83 + x34*x63;
      float x122 = -x121;
      float x123 = x111*x97 + x120*x93 + x122*x94;
      float x124 = -x102;
      float x125 = x100*x98;
      float x126 = x119 + x53*x89 + x59*x85;
      float x127 = x102*x126;
      float x128 = x100*x86 + x117*x118 + x123*x124 - 0.06*x125 - 0.06*x127;
      float x129 = dq[6] + x96;
      float x130 = -x129;
      float x131 = parms[67]*x130 + parms[68]*x109 + parms[69]*x117;
      float x132 = dq[6]*x109 + x125 + x127;
      float x133 = -parms[68];
      float x134 = parms[66]*x114 + parms[69]*x128 + x112*x113 + x129*x131 + x132*x133;
      float x135 = x100*x134;
      float x136 = -parms[67];
      float x137 = x100*x115 - 0.06*x103 - 0.06*x105 + x116*x124;
      float x138 = parms[66]*x129 + parms[69]*x137 + x106*x133;
      float x139 = x100*x126;
      float x140 = -0.06*x102;
      float x141 = dq[6]*x137 + x100*x123 + x102*x86 + 0.06*x139 + x140*x98;
      float x142 = x106*x118 + x124*x98 + x139;
      float x143 = parms[68]*x142 + parms[69]*x141 + x109*x112 + x114*x136 + x130*x138;
      float x144 = -parms[56];
      float x145 = parms[57]*x104 + parms[59]*x111 + x101*x144;
      float x146 = -x104;
      float x147 = -x96;
      float x148 = parms[57]*x147 + parms[58]*x101 + parms[59]*x116;
      float x149 = parms[47]*x65 + parms[49]*x86 + parms[56]*x114 + parms[59]*x86 + x102*x143 + x135 + x145*x146 + x148*x96 + x48*x49 + x77*x78 + x89*x92 + x98*x99;
      float x150 = x149*x40;
      float x151 = -x52;
      float x152 = -x109;
      float x153 = dq[5]*x116 + x120*x94 + x122*x95;
      float x154 = parms[56]*x96 + parms[59]*x115 + x104*x99;
      float x155 = -x148;
      float x156 = parms[57]*x98 + parms[59]*x153 + parms[67]*x132 + parms[69]*x153 + x101*x155 + x104*x154 + x106*x138 + x110*x142 + x126*x144 + x131*x152;
      float x157 = parms[47]*x89 + parms[49]*x115 + x49*x68;
      float x158 = x100*x143;
      float x159 = -parms[57];
      float x160 = parms[58]*x126 + parms[59]*x123 + x101*x145 + x114*x159 + x124*x134 + x147*x154 + x158;
      float x161 = parms[48]*x48 + parms[49]*x120 + x126*x66 + x151*x92 + x156*x94 + x157*x68 + x160*x93;
      float x162 = parms[37]*x67 + parms[38]*x37 + parms[39]*x69;
      float x163 = -parms[38];
      float x164 = parms[36]*x47 + parms[39]*x80 + x150 + x161*x50 + x162*x58 + x163*x59 + x28*x39;
      float x165 = -parms[26];
      float x166 = parms[26]*x18 + parms[28]*x82 + parms[29]*x30;
      float x167 = parms[27]*x43 + parms[29]*x79 + x164 + x165*x45 + x166*x25 + x20*x21;
      float x168 = -x69;
      float x169 = parms[32]*x27 + parms[34]*x37 + parms[35]*x58 + parms[36]*x74 + parms[37]*x168;
      float x170 = -x111;
      float x171 = parms[51]*x104 + parms[53]*x101 + parms[54]*x96 + parms[56]*x170 + parms[58]*x116;
      float x172 = parms[50]*x104 + parms[51]*x101 + parms[52]*x96 + parms[57]*x111 + x115*x99;
      float x173 = parms[61]*x106 + parms[63]*x109 + parms[64]*x129 + parms[68]*x117 + x110*x111;
      float x174 = parms[60]*x106 + parms[61]*x109 + parms[62]*x129 + parms[67]*x111 + x133*x137;
      float x175 = parms[62]*x132 + parms[64]*x142 + parms[65]*x114 + parms[66]*x128 + x106*x173 + x117*x138 - x131*x137 + x136*x141 + x152*x174;
      float x176 = parms[52]*x98 + parms[54]*x126 + parms[55]*x114 + parms[56]*x86 - x101*x172 + x104*x171 + x115*x155 + x116*x154 + x123*x159 + x175;
      float x177 = parms[42]*x89 + parms[44]*x68 + parms[45]*x52 + parms[46]*x90 + x66*x76;
      float x178 = parms[41]*x89 + parms[43]*x68 + parms[44]*x52 + parms[48]*x76 + x115*x49;
      float x179 = -x117;
      float x180 = parms[62]*x106 + parms[64]*x109 + parms[65]*x129 + parms[66]*x137 + parms[67]*x179;
      float x181 = parms[60]*x132 + parms[61]*x142 + parms[62]*x114 + parms[67]*x153 + x109*x180 + x112*x137 + x128*x133 + x130*x173 + x138*x170;
      float x182 = parms[61]*x132 + parms[63]*x142 + parms[64]*x114 + parms[68]*x141 + x110*x153 + x111*x131 + x112*x179 + x113*x180 + x129*x174;
      float x183 = -x116;
      float x184 = parms[52]*x104 + parms[54]*x101 + parms[55]*x96 + parms[56]*x115 + parms[57]*x183;
      float x185 = parms[50]*x98 + parms[51]*x126 + parms[52]*x114 + parms[57]*x153 + x100*x181 + x101*x184 + x115*x145 + x124*x182 - 0.06*x135 + x140*x143 + x147*x171 + x154*x170 + x86*x99;
      float x186 = parms[40]*x65 + parms[41]*x48 + parms[42]*x126 + parms[47]*x86 - x115*x92 + x122*x91 + x151*x178 + x157*x90 + x176*x94 + x177*x68 + x185*x93;
      float x187 = parms[31]*x27 + parms[33]*x37 + parms[34]*x58 + parms[36]*x90 + parms[38]*x69;
      float x188 = parms[40]*x89 + parms[41]*x68 + parms[42]*x52 + parms[47]*x115 + x90*x91;
      float x189 = parms[51]*x98 + parms[53]*x126 + parms[54]*x114 + parms[58]*x123 + x100*x182 + x102*x181 + x111*x148 + x134*x140 + x144*x153 + x145*x183 + x146*x184 + 0.06*x158 + x172*x96;
      float x190 = parms[42]*x65 + parms[44]*x48 + parms[45]*x126 + parms[46]*x122 + x120*x66 + x178*x89 + x188*x78 + x189 + x76*x92 - x77*x90;
      float x191 = parms[36]*x58 + parms[38]*x28 + parms[39]*x74;
      float x192 = parms[30]*x59 + parms[31]*x63 + parms[32]*x47 + parms[37]*x121 - 0.3*x150 + x161*x75 + x163*x80 + x169*x37 + x186*x40 + x187*x67 + x190*x85 + x191*x90 + x39*x74;
      float x193 = parms[22]*x25 + parms[24]*x16 + parms[25]*x18 + parms[26]*x30 + x14*x19;
      float x194 = parms[27]*x25 + parms[29]*x72 + x16*x165;
      float x195 = x164*x73;
      float x196 = -x89;
      float x197 = -parms[41]*x65 - parms[43]*x48 - parms[44]*x126 - parms[48]*x120 - x115*x77 + x157*x76 - x176*x95 - x177*x196 - x185*x94 - x188*x52 - x49*x86;
      float x198 = parms[30]*x27 + parms[31]*x37 + parms[32]*x58 + parms[37]*x36 + x163*x74;
      float x199 = parms[32]*x59 + parms[34]*x63 + parms[35]*x47 + parms[36]*x80 - parms[37]*x84 - x162*x74 + x187*x27 + x191*x69 + x197 + x198*x38;
      float x200 = -parms[28];
      float x201 = parms[20]*x25 + parms[21]*x16 + parms[22]*x18 + parms[27]*x72 + x200*x30;
      float x202 = -parms[21]*x43 - parms[23]*x45 - parms[24]*x63 - parms[28]*x81 + x14*x194 - x165*x79 - x18*x201 - x192*x22 - x193*x82 - x195*x31 - x199*x31 - x20*x72;
      float x203 = dq[1]*parms[13] + parms[11]*x6 + parms[14]*x2;
      float x204 = -x166;
      float x205 = x161*x40;
      float x206 = parms[37]*x48 + parms[38]*x63 + parms[39]*x84 + x149*x85 + x191*x67 + x205 + x37*x39;
      float x207 = -parms[36]*x63 + parms[37]*x59 + parms[39]*x121 - parms[46]*x126 - parms[49]*x122 - x156*x95 - x157*x196 - x160*x94 + x162*x38 + x191*x27 - x52*x77 - x65*x91;
      float x208 = parms[28]*x45 + parms[29]*x81 + x16*x194 + x18*x204 + x19*x63 + x206*x26 + x207*x22;
      float x209 = -0.55*x10;
      float x210 = parms[21]*x25 + parms[23]*x16 + parms[24]*x18 + parms[28]*x14 + x165*x72;
      float x211 = parms[20]*x43 + parms[21]*x45 + parms[22]*x63 + parms[27]*x79 + x16*x193 - x18*x210 + x192*x26 + x194*x30 + x195*x22 + x199*x22 + x200*x83 + x204*x72;
      float x212 = parms[31]*x59 + parms[33]*x63 + parms[34]*x47 + parms[36]*x122 + parms[38]*x84 + x149*x75 + x162*x36 + x168*x39 + x169*x28 + x186*x50 + x190*x40 + x198*x58 + 0.3*x205;
      float x213 = parms[22]*x43 + parms[24]*x45 + parms[25]*x63 + parms[26]*x83 + x14*x166 + x19*x81 - x20*x30 + x201*x21 + x207*x34 + x210*x25 + x212;
      float x214 = -x10;
      float x215 = x11*(parms[26]*x63 + parms[29]*x83 + x18*x20 + x194*x82 + x200*x43 + x206*x22 + x207*x31);
      float x216 = dq[1]*parms[14] + parms[12]*x6 + parms[15]*x2;
    //
      c_out[0] = -x1*(-dq[1]*x7 + parms[12]*x4 + parms[15]*x8 + x11*x167 + x202 + x203*x6) - x5*(dq[1]*x216 + parms[10]*x4 + parms[12]*x8 - 0.55*x167*x9 + x203*x3 + x208*x209 + x211*x9 + x213*x214 + x214*x215);
      c_out[1] = parms[11]*x4 + parms[14]*x8 + x10*x211 + x167*x209 + x2*x7 + 0.55*x208*x9 + x213*x9 + x215*x9 - x216*x6;
      c_out[2] = 0.045*x167 + x202;
      c_out[3] = 0.045*x207 + x212;
      c_out[4] = x197;
      c_out[5] = x189;
      c_out[6] = x175;
    //
      return;
    }


    void g( float* g_out, const float* parms, const float* q )
    {
      float x0 = sin(q[5]);
      float x1 = cos(q[5]);
      float x2 = cos(q[3]);
      float x3 = cos(q[1]);
      float x4 = 9.81*x3;
      float x5 = -x4;
      float x6 = -x5;
      float x7 = sin(q[1]);
      float x8 = -9.81*x7;
      float x9 = cos(q[2]);
      float x10 = x8*x9;
      float x11 = sin(q[3]);
      float x12 = x10*x11 + x2*x6;
      float x13 = -x12;
      float x14 = -x13;
      float x15 = sin(q[4]);
      float x16 = sin(q[2]);
      float x17 = -x16;
      float x18 = x17*x8;
      float x19 = x10*x2 + x11*x5;
      float x20 = cos(q[4]);
      float x21 = x15*x18 + x19*x20;
      float x22 = x0*x21 + x1*x14;
      float x23 = parms[59]*x22 + parms[69]*x22;
      float x24 = x0*x13 + x1*x21;
      float x25 = sin(q[6]);
      float x26 = -x25;
      float x27 = -x15;
      float x28 = x18*x20 + x19*x27;
      float x29 = cos(q[6]);
      float x30 = x24*x26 + x28*x29;
      float x31 = parms[69]*x30;
      float x32 = x24*x29 + x25*x28;
      float x33 = parms[69]*x32;
      float x34 = x29*x33;
      float x35 = parms[59]*x24 + x26*x31 + x34;
      float x36 = parms[49]*x21 + x0*x23 + x1*x35;
      float x37 = x15*x36;
      float x38 = x29*x31;
      float x39 = parms[49]*x28 + parms[59]*x28 + x25*x33 + x38;
      float x40 = parms[39]*x18 + x20*x39 + x37;
      float x41 = ((x11)*(x11));
      float x42 = ((x2)*(x2));
      float x43 = x40*(-0.045*x41 - 0.045*x42);
      float x44 = -x2;
      float x45 = -x18;
      float x46 = parms[67]*x22 - parms[68]*x30;
      float x47 = -x22;
      float x48 = parms[66]*x47 + parms[68]*x32;
      float x49 = -0.06*x25;
      float x50 = parms[56]*x47 + parms[58]*x24 + x25*x46 + x29*x48 + x31*x49 + 0.06*x34;
      float x51 = parms[46]*x13 - parms[47]*x21 + x50;
      float x52 = parms[66]*x30 - parms[67]*x32;
      float x53 = parms[56]*x28 - parms[57]*x24 + x52;
      float x54 = -x28;
      float x55 = parms[57]*x22 + parms[58]*x54 + x26*x48 + x29*x46 + x33*x49 - 0.06*x38;
      float x56 = parms[47]*x28 + parms[48]*x14 + x0*x53 + x1*x55;
      float x57 = -0.3*x39;
      float x58 = parms[37]*x12 + parms[38]*x45 + x20*x56 + x20*x57 + x27*x51 - 0.3*x37;
      float x59 = -x1;
      float x60 = -parms[46]*x54 - parms[48]*x21 - x0*x55 - x53*x59;
      float x61 = parms[36]*x18 - parms[37]*x19 + x60;
      float x62 = -parms[26]*x45 - parms[28]*x10 - x11*x58 - x43*x44 - x44*x61;
      float x63 = parms[29]*x18 + x40;
      float x64 = 0.045*((x16)*(x16)) + 0.045*((x9)*(x9));
      float x65 = x20*x36;
      float x66 = parms[36]*x13 + parms[38]*x19 + x15*x56 + x15*x57 + x20*x51 + 0.3*x65;
      float x67 = parms[39]*x12 - parms[49]*x13 - x0*x35 - x23*x59;
      float x68 = parms[26]*x5 - parms[27]*x10 + x66 + x67*(0.045*x41 + 0.045*x42);
      float x69 = parms[27]*x18 + parms[28]*x6 + x11*x43 + x11*x61 + x2*x58;
      float x70 = parms[39]*x19 + x27*x39 + x65;
      float x71 = parms[29]*x10 + x11*x67 + x2*x70;
      float x72 = -0.55*x16;
      float x73 = x64*(parms[29]*x5 + x11*x70 + x44*x67);
    //
      g_out[0] = x3*(-parms[17]*x8 + x62 + x63*x64) - x7*(parms[17]*x4 + x17*x68 + x17*x73 - 0.55*x63*x9 + x69*x9 + x71*x72);
      g_out[1] = parms[16]*x5 + parms[18]*x8 + x16*x69 + x63*x72 + x68*x9 + 0.55*x71*x9 + x73*x9;
      g_out[2] = x62 + 0.045*x63;
      g_out[3] = x66 + 0.045*x67;
      g_out[4] = x60;
      g_out[5] = x50;
      g_out[6] = x52;
    //
      return;
    }


    VectorXf continuous_dynamics(VectorXf x, VectorXf u, float weights[])
    {

        const int half_NX = NX/2;
        float max_control = 1.0;

        // Create Eigen data structures
        Matrix<float, half_NX, 1> tau;
        Matrix<float, NV, 1> virtual_control;
        Matrix<float, half_NX, 1> c_vec;
        Matrix<float, half_NX, 1> g_vec;
        Matrix<float, half_NX, half_NX> M_mat;
        Matrix<float, half_NX, 1> ddq; // For fully actuated system: NU = half_NX
        // Matrix<float, half_NX, 1> _ddq;

        // Create local arrays to pass into generated code
        float M_out[half_NX*half_NX], g_out[half_NX], c_out[half_NX], q[half_NX], dq[half_NX];

        // Fill in q and dq
        int idx = 0;
        for(int i = 0; i < half_NX; ++i) {
          dq[i] = x(idx++);
        }
        for(int i = 0; i < half_NX; ++i) {
          q[i] = x(idx++);
        }

        // Calculate using generated code
        M(M_out, weights, q);
        c(c_out, weights, q, dq);
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
        ddq = M_mat.lu().solve(max_control*tau - c_vec - g_vec)  + virtual_control.head(half_NX);
        // ddq = M_mat.inverse() * (tau - c_vec - g_vec);

        // Stack [ddq, dq] into xdot
        VectorXf xdot(NX);
        xdot.head(half_NX) << ddq;
        xdot.tail(half_NX) << x.head(half_NX) + virtual_control.tail(half_NX); // first half of x is dq

        return xdot;
    }

    MatrixXf numerical_jacobian(VectorXf (*f)(VectorXf, VectorXf, float[]), VectorXf x, VectorXf u, float delta, float weights[])
    {
        int nX = x.rows();
        int nU = u.rows();

        // Create matrix, set it to all zeros
        MatrixXf jac(nX, nX+nU+1);
        jac.setZero(nX, nX+nU+1);

        int index = 0;

        MatrixXf I;
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

    MatrixXf general_numerical_jacobian(VectorXf (*f)(VectorXf), VectorXf x, int out_dimension)
    {
        int nX = x.rows();

        // Create matrix, set it to all zeros
        MatrixXf jac(out_dimension, nX);
        jac.setZero(out_dimension, nX);

        int index = 0;

        MatrixXf I;
        I.setIdentity(nX, nX);
        for(int i = 0; i < nX; ++i) {
            jac.col(index) = f(x + .5*EPS*I.col(i)) - f(x - .5*EPS*I.col(i));
            index++;
        }

        // Must divide by eps for finite differences formula
        jac /= EPS;

        return jac;
    }

    VectorXf dynamics_difference(VectorXf (*f)(VectorXf, VectorXf, float[]), VectorXf x, VectorXf x_next, VectorXf u, float delta, float weights[])
    {
        VectorXf simulated_x_next = rk45_DP(f, x, u, delta, weights);
        return x_next - simulated_x_next;
    }

};

#endif /* PENDULUM_DYNAMICS_BY_HAND_H_ */

            
