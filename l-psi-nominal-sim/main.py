import numpy as np
import matplotlib.pyplot as plt
import functions
import constants
from scipy.integrate import odeint
import solver

tf = 10
N = 1000

#initial position, heading and velocity of follower
x_F1 = 0
x_F2 = 0
theta_F = 0
v_F = 1

# leader trajectory
def x_L1_traj(t):
    return np.cos(t)
def x_L1_dot_traj(t):
    return -np.sin(t)
def x_L1_ddot_traj(t):
    return -np.cos(t)
def x_L2_traj(t):
    return np.sin(t)
def x_L2_dot_traj(t):
    return np.cos(t)
def x_L2_ddot_traj(t):
    return -1*np.sin(t)
def theta_L_traj(t):
    return np.arctan2(x_L2_dot_traj(t), x_L1_dot_traj(t))
def theta_L_dot_traj(t):
    return 1
def theta_L_ddot_traj(t):
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
    return 0.5
def psi_des(time):
    return np.pi / 2
def l_dot_des(time):
    return 0
def psi_dot_des(time):
    return 0
def l_ddot_des(time):
    return  0
def psi_ddot_des(time):
    return 0

#desired output flat_space
def z_des_traj(time):
    psi_d = psi_des(time)
    psi_dot_d = psi_dot_des(time)
    psi_ddot_d = psi_ddot_des(time)
    l_d = l_des(time)
    l_dot_d = l_dot_des(time)
    l_ddot_d = l_ddot_des(time)
    theta_L = theta_L_traj(time)
    theta_L_dot = theta_L_dot_traj(time) 
    theta_L_ddot = theta_L_ddot_traj(time)
    zeta = np.pi - psi_d - theta_L
    y1_des = -1*l_d*np.cos(zeta)
    y2_des = l_d*np.sin(zeta)
    y1_dot_des =  -1*l_dot_d*np.cos(zeta) - l_d*np.sin(zeta)*(psi_dot_d + theta_L_dot)
    y2_dot_des = l_dot_d*np.sin(zeta) - l_d*np.cos(zeta)*(psi_dot_d + theta_L_dot)
    y1_ddot_des = -1*l_ddot_d*np.cos(zeta) - 2*l_dot_d*np.sin(zeta)*(psi_dot_d + theta_L_dot) + l_d*np.cos(zeta)*(psi_dot_d+theta_L_dot)**2 - l_d*np.sin(zeta)*(psi_ddot_d + theta_L_ddot)
    y2_ddot_des = l_ddot_d*np.sin(zeta) - 2*l_dot_d*np.cos(zeta)*(psi_dot_d + theta_L_dot) - l_d*np.sin(zeta)*(psi_dot_d + theta_L_dot)**2 - l_d*np.cos(zeta)*(psi_ddot_d + theta_L_ddot)
    return np.array([y1_des, y2_des, y1_dot_des, y2_dot_des, y1_ddot_des, y2_ddot_des])

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

    u = functions.flat_control_to_real_control(v, state, x_L_ddot) + constants.noise(time)

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

# plots
l_des_traj = [l_des(t) for t in time]
plt.plot(time, state_traj[:, 0], label=r'$l$')
plt.plot(time, l_des_traj, label=r"$l_d$")
plt.legend()
plt.ylabel(r"$l$ (meter)")
plt.xlabel("time (second)")
# plt.title(r"$l$ vs time")
plt.savefig("l.png")
plt.show()

psi_des_traj = [psi_des(t) for t in time]
plt.plot(time, state_traj[:, 1], label=r'$\psi$')
plt.plot(time, psi_des_traj, label=r"$\psi_d$")
plt.legend()
plt.ylabel(r"$\psi$ (radian)")
plt.xlabel("time (second)")
# plt.title(r"$\psi$ vs time")
plt.savefig("psi.png")
plt.show()

plt.plot(time, state_traj[:, 2], label=r'$\theta_F$')
plt.ylabel(r"$\theta_F$ (radian)")
plt.xlabel("time (second)")
# plt.title(r"$\theta_F$ vs time")
plt.savefig("theta_F.png")
plt.show()

plt.plot(time, state_traj[:, 3], label='v_F')
plt.ylabel(r"$v_F$ (meter/second)")
plt.xlabel("time (second)")
# plt.title(r"$v_F$ vs time")
plt.savefig("v_F.png")
plt.show()

indices = np.arange(0, N, 1)
number_arrows = 11
selected_indices = np.linspace(0, len(indices) - 1, number_arrows).astype(int)
arrow_points = indices[selected_indices]
arrow_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'black']

x_L1_data = np.array([x_L1_traj(t) for t in time])
x_L2_data = np.array([x_L2_traj(t) for t in time])
theta_L_data = np.array([theta_L_traj(t) for t in time])

plt.plot(state_traj[:, 4], state_traj[:, 5], label="follower")
plt.plot(x_L1_data, x_L2_data, label="leader")
for i, point in enumerate(arrow_points):
    plt.arrow(x_L1_data[point], x_L2_data[point], 0.1*np.cos(theta_L_data[point]), 0.1*np.sin(theta_L_data[point]),
              shape='full', lw=0.1, length_includes_head=True, head_width=.2, fc=arrow_colors[i], overhang=0.2)
    
    plt.arrow(state_traj[point, 4], state_traj[point, 5], 0.1*np.cos(state_traj[point, 2]), 0.1*np.sin(state_traj[point, 2]),
              shape='full', lw=0.1, length_includes_head=True, head_width=.2, fc=arrow_colors[i], overhang=0.2)

    # Add timestamp next to the arrow
    plt.text(x_L1_data[point], x_L2_data[point], f' t={time[point]:.2f}s', fontsize=8, ha='left', va='bottom', color='orange')
    plt.text(state_traj[point, 4], state_traj[point, 5], f't={time[point]:.2f}s', fontsize=8, ha='right', va='bottom', color='blue')


plt.legend()
plt.ylabel(r"$x$ (meter)")
plt.xlabel(r"$y$ (meter)")
plt.title(r"$xy$ plane trajectory")
plt.savefig("traj.png")
plt.show()






# l_des_traj = [l_des(t) for t in time]
# plt.figure(figsize=(6, 4)) 
# plt.subplot(2, 1, 1)
# plt.plot(time, state_traj[:, 0], label='l')
# plt.plot(time, l_des_traj, label="desired l ")
# plt.title('Simulating non-linear system - l')
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(time, state_traj_proj[:, 0], label='l')
# plt.plot(time, l_des_traj, label="desired l ")
# plt.title('Simulating flat system - l')
# plt.legend()
# plt.tight_layout() 
# plt.savefig("l.png")
# plt.show()

# psi_des_traj = [psi_des(t) for t in time]
# plt.figure(figsize=(6, 4)) 
# plt.subplot(2, 1, 1) 
# plt.plot(time, state_traj[:, 1], label='psi')
# plt.plot(time, psi_des_traj, label="desired psi ")
# plt.title('Simulating non-linear system - psi')
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(time, state_traj_proj[:, 1], label='psi')
# plt.plot(time, psi_des_traj, label="desired psi ")
# plt.title('Simulating flat system - psi')
# plt.legend()
# plt.tight_layout() 
# plt.savefig("psi.png")
# plt.show()

# plt.figure(figsize=(6, 4)) 
# plt.subplot(2, 1, 1) 
# plt.plot(time, state_traj[:, 2], label='theta_F')
# plt.title('Simulating non-linear system - theta_F')
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(time, state_traj_proj[:, 2], label='theta_F')
# plt.title('Simulating flat system - theta_F')
# plt.legend()
# plt.tight_layout() 
# plt.savefig("theta_F.png")
# plt.show()

# plt.figure(figsize=(6, 4)) 
# plt.subplot(2, 1, 1) 
# plt.plot(time, state_traj[:, 3], label='v_F')
# plt.title('Simulating non-linear system - v_F')
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(time, state_traj_proj[:, 3], label='v_F')
# plt.title('Simulating flat system - v_F')
# plt.legend()
# plt.tight_layout() 
# plt.savefig("v_F.png")
# plt.show()

# plt.plot(state_traj[:, 4], state_traj[:, 5], "-o", label="follower")
# x_L1_data = np.array([x_L1_traj(t) for t in time])
# x_L2_data = np.array([x_L2_traj(t) for t in time])
# plt.plot(x_L1_data, x_L2_data, "-o", label="leader")
# plt.title("leader follower trajectory")
# plt.legend()
# plt.savefig("traj.png")
# plt.show()