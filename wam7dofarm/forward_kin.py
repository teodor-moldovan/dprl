import numpy as np
from numpy import cos, sin

def end_effector_pos(x):

    dq = x[:7]
    q = x[7:]

    s1 = sin(q[0]);
    s2 = sin(q[1]);
    s3 = sin(q[2]);
    s4 = sin(q[3]);
    s5 = sin(q[4]);
    s6 = sin(q[5]);
    s7 = sin(q[6]);

    c1 = cos(q[0]);
    c2 = cos(q[1]);
    c3 = cos(q[2]);
    c4 = cos(q[3]);
    c5 = cos(q[4]);
    c6 = cos(q[5]);
    c7 = cos(q[6]);

    return np.matrix([
    [0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 - 0.045*(-s1*s3 + c1*c2*c3)*c4 - 0.045*s1*s3 + 0.045*s2*s4*c1 + 0.3*s2*c1*c4 + 0.55*s2*c1 + 0.045*c1*c2*c3],
    [    0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 - 0.045*(s1*c2*c3 + s3*c1)*c4 + 0.045*s1*s2*s4 + 0.3*s1*s2*c4 + 0.55*s1*s2 + 0.045*s1*c2*c3 + 0.045*s3*c1],
    [                                                                                                                                                                                                                         0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.045*s2*c3*c4 - 0.045*s2*c3 + 0.045*s4*c2 + 0.3*c2*c4 + 0.55*c2]])

    
def pos_jac(x):

    q = x[7:]

    s1 = sin(q[0]);
    s2 = sin(q[1]);
    s3 = sin(q[2]);
    s4 = sin(q[3]);
    s5 = sin(q[4]);
    s6 = sin(q[5]);
    s7 = sin(q[6]);

    c1 = cos(q[0]);
    c2 = cos(q[1]);
    c3 = cos(q[2]);
    c4 = cos(q[3]);
    c5 = cos(q[4]);
    c6 = cos(q[5]);
    c7 = cos(q[6]);

    J = np.matrix([
    [   -0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 + 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 - 0.3*(s1*c2*c3 + s3*c1)*s4 + 0.045*(s1*c2*c3 + s3*c1)*c4 - 0.045*s1*s2*s4 - 0.3*s1*s2*c4 - 0.55*s1*s2 - 0.045*s1*c2*c3 - 0.045*s3*c1,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        (0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.045*s2*c3*c4 - 0.045*s2*c3 + 0.045*s4*c2 + 0.3*c2*c4 + 0.55*c2)*c1,                                                                                                                                                                                                                                   (0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.045*s2*c3*c4 - 0.045*s2*c3 + 0.045*s4*c2 + 0.3*c2*c4 + 0.55*c2)*s1*s2 - (0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 - 0.045*(s1*c2*c3 + s3*c1)*c4 + 0.045*s1*s2*s4 + 0.3*s1*s2*c4 + 0.55*s1*s2 + 0.045*s1*c2*c3 + 0.045*s3*c1)*c2,                                                                                                                                                                                                               (-s1*s3*c2 + c1*c3)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.045*s2*c3*c4 + 0.045*s4*c2 + 0.3*c2*c4) - (0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 - 0.045*(s1*c2*c3 + s3*c1)*c4 + 0.045*s1*s2*s4 + 0.3*s1*s2*c4)*s2*s3,                                                                                                                                                                                 ((s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.3*c2*c4) - (-s2*s4*c3 + c2*c4)*(0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 + 0.3*s1*s2*c4),                                                                                                                                                                         -(0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6)*(-(-s2*c3*c4 - s4*c2)*s5 + s2*s3*c5) + (-((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*s5 + (-s1*s3*c2 + c1*c3)*c5)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6),                                                                                                                                                                                                               -(0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6)*(((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - (s2*s4*c3 - c2*c4)*c6) + ((((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - (-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6)],
    [0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 - 0.045*(-s1*s3 + c1*c2*c3)*c4 - 0.045*s1*s3 + 0.045*s2*s4*c1 + 0.3*s2*c1*c4 + 0.55*s2*c1 + 0.045*c1*c2*c3,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        (0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.045*s2*c3*c4 - 0.045*s2*c3 + 0.045*s4*c2 + 0.3*c2*c4 + 0.55*c2)*s1,                                                                                                                                                                                                                              -(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.045*s2*c3*c4 - 0.045*s2*c3 + 0.045*s4*c2 + 0.3*c2*c4 + 0.55*c2)*s2*c1 + (0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 - 0.045*(-s1*s3 + c1*c2*c3)*c4 - 0.045*s1*s3 + 0.045*s2*s4*c1 + 0.3*s2*c1*c4 + 0.55*s2*c1 + 0.045*c1*c2*c3)*c2,                                                                                                                                                                                                          -(-s1*c3 - s3*c1*c2)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.045*s2*c3*c4 + 0.045*s4*c2 + 0.3*c2*c4) + (0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 - 0.045*(-s1*s3 + c1*c2*c3)*c4 + 0.045*s2*s4*c1 + 0.3*s2*c1*c4)*s2*s3,                                                                                                                                                                            -((-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6 - 0.3*s2*s4*c3 + 0.3*c2*c4) + (-s2*s4*c3 + c2*c4)*(0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 + 0.3*s2*c1*c4),                                                                                                                                                                       (0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6)*(-(-s2*c3*c4 - s4*c2)*s5 + s2*s3*c5) - (-((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*s5 + (-s1*c3 - s3*c1*c2)*c5)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6),                                                                                                                                                                                                            (0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6)*(((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - (s2*s4*c3 - c2*c4)*c6) - ((((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - (-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6)*(0.06*((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - 0.06*(s2*s4*c3 - c2*c4)*c6)],
    [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   0, -(0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 - 0.045*(-s1*s3 + c1*c2*c3)*c4 - 0.045*s1*s3 + 0.045*s2*s4*c1 + 0.3*s2*c1*c4 + 0.55*s2*c1 + 0.045*c1*c2*c3)*c1 - (0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 - 0.045*(s1*c2*c3 + s3*c1)*c4 + 0.045*s1*s2*s4 + 0.3*s1*s2*c4 + 0.55*s1*s2 + 0.045*s1*c2*c3 + 0.045*s3*c1)*s1, -(0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 - 0.045*(-s1*s3 + c1*c2*c3)*c4 - 0.045*s1*s3 + 0.045*s2*s4*c1 + 0.3*s2*c1*c4 + 0.55*s2*c1 + 0.045*c1*c2*c3)*s1*s2 + (0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 - 0.045*(s1*c2*c3 + s3*c1)*c4 + 0.045*s1*s2*s4 + 0.3*s1*s2*c4 + 0.55*s1*s2 + 0.045*s1*c2*c3 + 0.045*s3*c1)*s2*c1, (-s1*c3 - s3*c1*c2)*(0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 - 0.045*(s1*c2*c3 + s3*c1)*c4 + 0.045*s1*s2*s4 + 0.3*s1*s2*c4) - (-s1*s3*c2 + c1*c3)*(0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 - 0.045*(-s1*s3 + c1*c2*c3)*c4 + 0.045*s2*s4*c1 + 0.3*s2*c1*c4), ((-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*(0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6 + 0.3*(s1*c2*c3 + s3*c1)*s4 + 0.3*s1*s2*c4) - ((s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*(0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6 + 0.3*(-s1*s3 + c1*c2*c3)*s4 + 0.3*s2*c1*c4), -(0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6)*(-((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*s5 + (-s1*s3*c2 + c1*c3)*c5) + (0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6)*(-((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*s5 + (-s1*c3 - s3*c1*c2)*c5), -(0.06*(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - 0.06*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6)*((((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - (-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6) + ((((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - (-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4)*c6)*(0.06*(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - 0.06*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4)*c6)],
    ])

    return J

def end_effector_lin_vel(x):

    dq = x[:7]
    dq = np.matrix(dq.reshape(-1)).T    
 
    J = pos_jac(x)

    lin_vel = J*dq

    return lin_vel[:3]

def p(x):

    pos = end_effector_pos(x)
    vel = end_effector_lin_vel(x)
    return np.concatenate((vel, pos))
 
def numerical_jacobian(f, x, out_dimension):

    nX = x.shape[0]
    jac = np.zeros((out_dimension, nX))
    eps = 1e-5
    for i in range(nX):
        temp_up = x.copy()
        temp_down = x.copy()
        temp_up[i] += .5 * eps
        temp_down[i] -= .5 * eps
        jac[:,i] = np.array((f(temp_up) - f(temp_down)).reshape(-1))[0]
    jac /= eps

    return jac

def p_jac(x):

    def top_right_corner_numerical_jac(x):

        trc_jac = np.zeros((3,7))
        eps = 1e-5
        for i in range(7):
            temp_up = x.copy()
            temp_down = x.copy()
            temp_up[i+7] += .5 * eps
            temp_down[i+7] -= .5 * eps
            trc_jac[:,i] = np.array((end_effector_lin_vel(temp_up) - end_effector_lin_vel(temp_down)).reshape(-1))[0]
        trc_jac /= eps

        return trc_jac

    jac = np.zeros((6,14)) # hard coded dimensions for this specific problem
    temp = pos_jac(x)
    jac[:3, :7] = temp
    jac[3:, 7:] = temp
    jac[:3, 7:] = top_right_corner_numerical_jac(x)

    return jac
    
