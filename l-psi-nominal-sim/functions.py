import numpy as np
import constants

def l_psi_dynamics(state, u, state_L, state_L_dot):
    l = state[0]
    psi = state[1]
    theta_F = state[2]
    v_F = state[3]

    theta_L = state_L[2]
    v_L = state_L[3]
    theta_L_dot = state_L_dot[2]

    gamma = psi + theta_L - theta_F

    l_dot = v_F*np.cos(gamma) - v_L*np.cos(psi)
    psi_dot = (v_L*np.sin(psi) - v_F*np.sin(gamma) - theta_L_dot*l) / (l)
    theta_F_dot = (v_F * u[0] / constants.L)
    v_F_dot = u[1]
    x_F1_dot = v_F * np.cos(theta_F)
    x_F2_dot = v_F*np.sin(theta_F)

    return np.array([l_dot, psi_dot, theta_F_dot, v_F_dot, x_F1_dot, x_F2_dot])

def flat_to_nonlinear(y, y_dot, state_L, state_L_dot):
    theta_L = state_L[2]
    x_L1_dot = state_L_dot[0]
    x_L2_dot = state_L_dot[1]

    l = np.linalg.norm(y)
    psi = np.pi - np.arctan2(y[1], -1*y[0]) - theta_L
    theta_F = np.arctan2((y_dot[1] + x_L2_dot), (y_dot[0] + x_L1_dot))
    v_F = ((y_dot[0] + x_L1_dot)**2 + (y_dot[1] + x_L2_dot)**2)**0.5
    return np.array([l, psi, theta_F, v_F])

def nonlinear_to_flat(state, state_L, state_L_dot):
    l = state[0]
    psi = state[1]
    theta_F = state[2]
    v_F = state[3]
    theta_L = state_L[2]

    x_L1_dot = state_L_dot[0]
    x_L2_dot = state_L_dot[1]

    y_1 = -1*l*np.cos(np.pi - psi - theta_L)
    y_2 = l*np.sin(np.pi - psi - theta_L)
    y_1_dot = v_F*np.cos(theta_F) - x_L1_dot
    y_2_dot = v_F*np.sin(theta_F) - x_L2_dot

    return np.array([y_1, y_2, y_1_dot, y_2_dot])

def flat_system_dynamics(z, v):
    y = z[0:2]
    y_dot = z[2:4]

    z1_dot = y_dot
    z2_dot = v

    return np.concatenate([z1_dot, z2_dot])

def pd_controller(z, z_des):
    y = z[0:2]
    y_dot = z[2:4]

    y_des = z_des[0:2]
    y_dot_des = z_des[2:4]
    y_ddot_des = z_des[4:6]

    v = y_ddot_des + constants.Kp*(y_des - y) + constants.Kd*(y_dot_des - y_dot)
    return v

def flat_control_to_real_control(v, state, x_L_ddot):
    theta_F = state[2]
    v_F = state[3]

    x_L1_dot_dot = x_L_ddot[0]
    x_L2_dot_dot = x_L_ddot[1]

    v_1 = v[0]
    v_2 = v[1]

    u_1 = (constants.L / v_F**2) * (-1*(v_1 + x_L1_dot_dot)*np.sin(theta_F) + (v_2 + x_L2_dot_dot)*np.cos(theta_F))
    u_2 = (v_1 + x_L1_dot_dot)*np.cos(theta_F) + (v_2 + x_L2_dot_dot)*np.sin(theta_F)

    return np.array([u_1, u_2])

# def find_flat_traj(des_traj, state_L, state_L_dot):
#     l_d = des_traj[0]
#     psi_d = des_traj[1]

#     theta_L = state_L[2]

#     l_d_dot = des_traj[2]
#     psi_dot = des_traj[3]

#     theta_L_dot = state_L_dot[2]

#     y_d1 = -1*l_d * np.cos(np.pi - psi_d - theta_L)
#     y_d2 = l_d * np.sin(np.pi - psi_d - theta_L)

#     A = np.array([[y_d2, y_d1], [-1*y_d1, y_d2]])
#     b = np.array([l_d_dot/l_d, -1*psi_dot - theta_L_dot])
#     y_d1_dot, y_d2_dot = A.dot(b)

#     return np.array([y_d1, y_d2, y_d1_dot, y_d2_dot])


