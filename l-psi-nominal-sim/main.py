import numpy as np
import matplotlib.pyplot as plt
import constants

def l_psi_dynamics(state_F, u, state_L, state_L_dot):
    l = state_F[0]
    psi = state_F[1]
    theta_F = state_F[2]
    v_F = state_F[3]

    theta_L = state_L[2]
    v_L = state_L[3]
    theta_L_dot = state_L_dot[2]

    gamma = psi + theta_L - theta_F

    l_dot = v_F*np.cos(gamma) - v_L*np.cos(psi)
    psi_dot = (v_L*np.sin(psi) - v_F*np.sin(gamma) - theta_L_dot*l) / (l)
    theta_F_dot = (v_F * u[0] / constants.L)
    v_F_dot = u[1]

    return np.array([l_dot, psi_dot, theta_F_dot, v_F_dot])

def flat_to_nonlinear(y, y_dot, state_L, state_L_dot):
    theta_L = state_L[2]
    x_L1_dot = state_L_dot[1]
    x_L2_dot = state_L_dot[2]

    l = np.linalg.norm(y)
    psi = np.pi + np.arctan2(y[1], y[0]) - theta_L
    theta_F = np.arctan2((y_dot[1] + x_L2_dot), (y_dot[0], x_L1_dot))
    v_F = ((y_dot[0] + x_L1_dot)**2 + (y_dot[1] + x_L2_dot)**2)**0.5
    return np.array([l, psi, theta_F, v_F])


