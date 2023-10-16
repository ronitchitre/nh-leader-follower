import numpy as np
import matplotlib.pyplot as plt
import functions
import constants
from scipy.integrate import odeint
import solver

tf = 20
N = 1000

#initial position, heading and velocity of follower
x_F1 = 0
x_F2 = 0
theta_F = 0
v_F = 1

# leader trajectory
def x_L1_traj(t):
    return 1
def x_L1_dot_traj(t):
    return 0
def x_L1_ddot_traj(t):
    return 0
def x_L2_traj(t):
    return t+1
def x_L2_dot_traj(t):
    return 1
def x_L2_ddot_traj(t):
    return 0
def theta_L_traj(t):
    return np.pi / 2
def theta_L_dot_traj(t):
    return 0
def v_L_traj(t):
    return 1
def v_L_dot_traj(t):
    return 0

#initial flat output
y1 = x_F1 - x_L1_traj(0)
y2 = x_F2 - x_L2_traj(0)
y1_dot = v_F*np.cos(theta_F) - x_L1_dot_traj(0)
y2_dot = v_F*np.sin(theta_F) - x_L2_dot_traj(0)
z = np.array([y1, y2, y1_dot, y2_dot])

#initial state
l = ((x_F1 - x_L1_traj(0))**2 + (x_F2 - x_L2_traj(0))**2)**(0.5)
psi = np.pi - np.arctan2(x_F2 - x_L2_traj(0), -1*x_F1 + x_L1_traj(0)) - theta_L_traj(0)
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
    y1_des = -1*l_des(time)*np.cos(np.pi - psi_des(time) - theta_L_traj(time))
    y2_des = l_des(time)*np.sin(np.pi - psi_des(time) - theta_L_traj(time))
    b2 = l_dot_des(time) / l_des(time)
    b1 = psi_dot_des(time) + theta_L_dot_traj(time)
    y1_dot_des =  y2_des*b1 + y1_des*b2
    y2_dot_des = -1*y1_des*b1 + y2_des*b2
    return np.array([y1_des, y2_des, y1_dot_des, y2_dot_des])

def system_dynamics_flat(t, z):
    z_des = z_des_traj(t)
    v = functions.pd_controller(z, z_des)
    z_dot = functions.flat_system_dynamics(z, v)
    return z_dot

time = np.linspace(0, tf, N)
z_traj = solver.rk4method(system_dynamics_flat, z, time)

def v_traj(time):
    z_des = z_des_traj(time)
    i = int((N - 1)*time / tf) #check if correct
    z = z_traj[i]
    return functions.pd_controller(z, z_des)

def system_dynamics_nh(time, state):
    # v = v_traj(time)
    x_L_ddot = np.array([x_L1_ddot_traj(time), x_L2_ddot_traj(time)])
    x_L1 = x_L1_traj(time)
    x_L2 = x_L2_traj(time)
    theta_L = theta_L_traj(time)
    v_L = v_L_traj(time)
    state_L = np.array([x_L1, x_L2, theta_L, v_L])

    x_L1_dot = x_L1_dot_traj(time)
    x_L2_dot = x_L2_dot_traj(time)
    theta_L_dot = theta_L_dot_traj(time)
    v_L_dot = v_L_dot_traj(time)
    state_L_dot = np.array([x_L1_dot, x_L2_dot, theta_L_dot, v_L_dot])

    z = functions.nonlinear_to_flat(state, state_L, state_L_dot)
    # z = np.array([x_F1 - x_L1, x_F2 - x_L2, v_F*np.cos(theta_F) - x_L1_dot, v_F*np.sin(theta_F) - x_L2_dot])
    v = functions.pd_controller(z, z_des=z_des_traj(time))

    # u1 = constants.L*((z[2]+x_L1_dot)*(v[1]+x_L_ddot[1]) - (z[3]+x_L2_dot)*(v[0]+x_L_ddot[0])) / (v_F**3)
    # u2 = ((z[2]+x_L1_dot)*(v[0]+x_L_ddot[0]) + (z[3]+x_L2_dot)*(v[1]+x_L_ddot[1])) / v_F
    # u = np.array([u1, u2])

    u = functions.flat_control_to_real_control(v, state, x_L_ddot)

    state_dot = functions.l_psi_dynamics(state, u, state_L, state_L_dot)
    return state_dot

def follower_dynamics(time, state):
    # v = v_traj(time)
    x_L_ddot = np.array([x_L1_ddot_traj(time), x_L2_ddot_traj(time)])

    x_L1 = x_L1_traj(time)
    x_L2 = x_L2_traj(time)
    theta_L = theta_L_traj(time)
    v_L = v_L_traj(time)
    state_L = np.array([x_L1, x_L2, theta_L, v_L])

    x_L1_dot = x_L1_dot_traj(time)
    x_L2_dot = x_L2_dot_traj(time)
    theta_L_dot = theta_L_dot_traj(time)
    v_L_dot = v_L_dot_traj(time)
    state_L_dot = np.array([x_L1_dot, x_L2_dot, theta_L_dot, v_L_dot])

    x_F1 = state[0]
    x_F2 = state[1]
    theta_F = state[2]
    v_F = state[3]

    z = np.array([x_F1 - x_L1, x_F2-x_L2, v_F*np.cos(theta_F)-x_L1_dot, v_F*np.sin(theta_F)-x_L2_dot])
    v = functions.pd_controller(z, z_des=z_des_traj(time))

    u1 = constants.L*((z[2]+x_L1_dot)*(v[1]+x_L_ddot[1]) - (z[3]+x_L2_dot)*(v[0]+x_L_ddot[0])) / (v_F**3)
    u2 = ((z[2]+x_L1_dot)*(v[0]+x_L_ddot[0]) + (z[3]+x_L2_dot)*(v[1]+x_L_ddot[1])) / v_F

    state_dot = np.array([v_F*np.cos(theta_F), v_F*np.sin(theta_F), v_F*u1/constants.L, u2])
    return state_dot

state_ic = np.array([l, psi, theta_F, v_F, x_F1, x_F2])
state_traj = solver.rk4method(system_dynamics_nh, state_ic, time)

state_traj_proj = np.zeros((state_traj.shape[0], 4))
i = 0
for t in time:
    state_L = np.array([x_L1_traj(t), x_L2_traj(t), theta_L_traj(t), v_L_traj(t)])
    state_L_dot = np.array([x_L1_dot_traj(t), x_L2_dot_traj(t), theta_L_dot_traj(t), v_L_dot_traj(t)])
    z = z_traj[i]
    state_traj_proj[i] = functions.flat_to_nonlinear(z[0:2], z[2:4], state_L, state_L_dot)
    i+=1

#plots
plt.plot(time, state_traj[:,0], label="l")
l_des_traj = [l_des(t) for t in time]
plt.plot(time, l_des_traj, label="l desired")
plt.legend()
plt.show()

plt.plot(time, state_traj[:,1]*180/np.pi, label="psi")
psi_des_traj = [psi_des(t)*180/np.pi for t in time]
plt.plot(time, psi_des_traj, label="psi desired")
plt.legend()
plt.show()

plt.plot(time, state_traj[:,2], label="theta_F")
plt.legend()
plt.show()

plt.plot(time, state_traj[:,3], label="v_F")
plt.legend()
plt.show()

plt.plot(time, state_traj_proj[:,0], label="l")
l_des_traj = [l_des(t) for t in time]
plt.plot(time, l_des_traj, label="l desired")
plt.legend()
plt.title("proj")
plt.show()

# plt.plot(time, 180 - (state_traj_proj[:,1]*180/np.pi) - 90, label="psi")
# psi_des_traj = [180 - (psi_des(t)*180/np.pi) - 90 for t in time]
# plt.plot(time, psi_des_traj, label="psi desired")
# plt.legend()
# plt.title("actual psi")
# plt.show()

# plt.plot(time, state_traj_proj[:, 3], label="v_F")
# plt.legend()
# plt.show()

# plt.plot(time, z_traj[:,0], label="y1")
# y1_des_traj = [z_des_traj(t)[0] for t in time]
# plt.plot(time, y1_des_traj, label="y1 desired")
# plt.legend()
# plt.show()

# plt.plot(time, z_traj[:,1], label="y2")
# y2_des_traj = [z_des_traj(t)[1] for t in time]
# plt.plot(time, y2_des_traj, label="y2 desired")
# plt.legend()
# plt.show()

# plt.plot(time, z_traj[:,2], label="y1_dot")
# y1_dot_des_traj = [z_des_traj(t)[2] for t in time]
# plt.plot(time, y1_dot_des_traj, label="y1_dot desired")
# plt.legend()
# plt.show()

# plt.plot(time, z_traj[:,3], label="y2_dot")
# y2_dot_des_traj = [z_des_traj(t)[3] for t in time]
# plt.plot(time, y2_dot_des_traj, label="y2_dot desired")
# plt.legend()
# plt.show()

# plt.plot(z_traj[:, 0], z_traj[:, 1], label="y")
# plt.plot(y1_des_traj, y2_des_traj, label="desired y")
# plt.legend()
# plt.show()

x_L1_traj = np.array([x_L1_traj(t) for t in time])
x_F1_traj = z_traj[:, 0] + x_L1_traj
x_L2_traj = np.array([x_L2_traj(t) for t in time])
x_F2_traj = z_traj[:, 1] + x_L2_traj
plt.plot(x_F1_traj, x_F2_traj, "-o", label="follow traj")
plt.plot(x_L1_traj, x_L2_traj, "-o", label="leader traj")
plt.legend()
plt.show()

x_L1_dot_traj = np.array([x_L1_dot_traj(t) for t in time])
x_F1_dot_traj = z_traj[:, 2] + x_L1_dot_traj
x_L2_dot_traj = np.array([x_L2_dot_traj(t) for t in time])
x_F2_dot_traj = z_traj[:, 3] + x_L2_dot_traj
plt.plot((x_F1_dot_traj**2 + x_F2_dot_traj**2)**0.5, label="v_dot_F")
plt.legend()
plt.show()

plt.plot(state_traj[:, 4], state_traj[:, 5], "-o", label="follower")
plt.plot(x_F1_traj, x_F2_traj, "-o", label="projected")
plt.legend()
plt.show()