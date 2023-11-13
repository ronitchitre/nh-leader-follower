import numpy as np
from random import uniform

L = 1
Kp = 1
Kd = 2

def atan2(x1, x2):
    return np.arctan2(x1, x2) + np.pi

def noise(t):
    lmda1 = np.random.normal(0, 0.1)
    lmda2 = np.random.normal(0, 0.1)
    return np.array([0, 0])