#!/usr/bin/python3

from math import pi, sin, cos, sqrt, log
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

def f_xy(x, y):
    return 5 * pi**2 * sin(pi * (x - 1)) * sin(2 * pi * (y - 1))

def u_xy(x, y):
    return sin(pi * (x - 1)) * sin(2 * pi * (y - 1))

def Ax(u):
    dim = u.shape[0]
    m = int(sqrt(u.shape[0]))
    h = 1. / (m + 1)
    Au = np.zeros(shape = u.shape)
    for k in range(0, m * m):
        i = k % m
        j = int(k / m)
        Au[k] = 20. * u[k] - 4. * (
                (u[k + 1] if i != m - 1 else 0) +
                (u[k - 1] if i != 0 else 0) +
                (u[k + m] if j != m - 1 else 0) +
                (u[k - m] if j != 0 else 0)) - (
                (u[k + m + 1] if i != m - 1 and j != m - 1 else 0) +
                (u[k + m - 1] if i != 0 and j != m - 1 else 0) +
                (u[k - m + 1] if i != m - 1 and j != 0 else 0) +
                (u[k - m - 1] if i != 0 and j != 0 else 0))
    return Au / (6 * h**2)

points = [50, 100, 150, 200, 250, 300]
logh = np.ndarray(shape = (len(points), 1))
loge = np.ndarray(shape = (len(points), 1))
for a in range(len(points)):
    m = points[a]
    h = 1. / (m + 1)
    logh[a] = log(h)
    f = np.zeros(shape = (m**2, 1))
    u_actual = np.zeros(shape = (m**2, 1))
    u = np.zeros(shape = (m**2, 1))
    for i in range(0, m):
        for j in range(0, m):
            f[i + m * j] = f_xy((i + 1) * h, (j + 1) * h)
            u_actual[i + m * j] = u_xy((i + 1) * h, (j + 1) * h)
    Au = Ax(u)
    r = f - Au
    p = r
    threshold = 1e-5
    while norm(r) > threshold:
        
        w = Ax(p)
        alpha = np.asscalar(norm(r)**2 / np.dot(p.transpose(),w))
        u = u + alpha * p
        norm_r_prev = norm(r)
        r = r - alpha * w
        beta = norm(r)**2 / norm_r_prev**2
        p = r + beta * p
    loge[a] = log(norm(u_actual - u, np.inf))
p, K = np.linalg.lstsq(np.stack([logh.ravel(), np.ones(len(points))]).T, loge.ravel())[0]
print(p, K)
plt.plot(logh, loge)
plt.show()
