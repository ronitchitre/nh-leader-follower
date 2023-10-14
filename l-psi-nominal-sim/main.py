import numpy as np
import matplotlib.pyplot as plt
import functions
from scipy.integrate import odeint

tf = 20
N = 1000

#initial position, heading and velocity of follower
x_F1 = 0
x_F2 = 0
theta_F = np.pi / 2
v_F = 1

# leader trajectory
def x_L1_traj(t):
    return 1
def x_L1_dot_traj(t):
    return 0
def x_L1_ddot_traj(t):
    return 0
def x_L2_traj(t):
    return 1 + t
def x_L2_dot_traj(t):
    return 1
def x_L2_ddot_traj(t):
    return 0
def theta_L_traj(t):
    return np.pi/2
def theta_L_dot_traj(t):
    return 0
def v_L_traj(t):
    return 1
def v_L_dot_traj(t):
    return 0

#initial flat output
y1 = x_F1 - x_L1_dot_traj(0)
y2 = x_F2 - x_L2_dot_traj(0)
y1_dot = v_F*np.cos(theta_F) - x_L1_dot_traj(0)
y2_dot = v_F*np.sin(theta_F) - x_L2_dot_traj(0)
z = np.array([y1, y2, y1_dot, y2_dot])

#initial state
l = ((x_F1 - x_L1_traj(0))**2 + (x_F2 - x_L2_traj(0))**2)**0.5
psi = np.pi + np.arctan2(x_L2_traj(0) - x_F2, x_L1_traj(0) - x_F1) - theta_L_traj(0)
state_ic = np.array([l, psi, theta_F, v_F])

#desired output
def l_des(time):
    return 1
def psi_des(time):
    return 0
def l_dot_des(time):
    return 0
def psi_dot_des(time):
    return 0

#desired output flat_space
def z_des_traj(time):
    y1_des = l_des(time)*np.cos(psi_des(time) + theta_L_traj(time) - np.pi)
    y2_des = l_des(time)*np.sin(psi_des(time) + theta_L_traj(time) - np.pi)
    b1 = l_dot_des(time) / l_des(time)
    b2 = psi_dot_des(time) + theta_L_dot_traj(time)
    y1_dot_des =  y1_des*b1 - y2_des*b2
    y2_dot_des = y2_des*b1 + y1_des*b2
    return np.array([y1_des, y2_des, y1_dot_des, y2_dot_des])

def system_dynamics_flat(z, t):
    z_des = z_des_traj(t)
    v = functions.pd_controller(z, z_des)
    z_dot = functions.flat_system_dynamics(z, v)
    return z_dot

time = np.linspace(0, tf, N)
z_traj = odeint(system_dynamics_flat, z, time)

def v_traj(time):
    z_des = z_des_traj(time)
    i = (N - 1)*time / tf #check if correct
    z = z_traj[i]
    return functions.pd_controller(z, z_des)

def system_dynamics_nh(state, time):
    v = v_traj(time)
    x_L_ddot = np.array([x_L1_ddot_traj(time), x_L2_ddot_traj(time)])
    u = functions.flat_control_to_real_control(v, state, x_L_ddot)
    state_dot = functions.l_psi_dynamics(state, u)

state_traj = odeint(system_dynamics_nh, state_ic, time)

state_traj_proj = np.zeros_like(state_traj)
for t in time:
    state_L = np.array([x_L1_traj(t), x_L2_traj(t), theta_L_traj(t), v_L_traj(t)])
    state_L_dot = np.array([x_L1_dot_traj(t), x_L2_dot_traj(t), theta_L_dot_traj(t), v_L_dot_traj(t)])
    i = (N - 1)*time / tf
    z = z_traj[i]
    state_traj_proj[i] = functions.flat_to_nonlinear(z[0:2], z[2:4], state_L, state_L_dot)

