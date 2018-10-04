#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

error = np.zeros(4)
i = 0
for n in [499, 199, 49, 9]:
	h = 1. / (n+1)
	ones = np.ones(n+1)
	data = np.vstack([ones/12., ones*(-4.)/3, ones*(h*h+2.5),\
		ones*(-4.)/3, ones/(12.)])
	A = spdiags(data, range(-2,3), n+1, n+1).toarray()
	A[0][n-1], A[0][n] = [1./12, -4./3]
	A[1][n] = 1./12
	A[n-1][0] = 1./12
	A[n][0], A[n][1] = [-4./3, 1./12]
	f = h*h*np.sin(4*np.pi * np.array(range(n+1)).T * h)
	u = np.linalg.solve(A,f)
	x = np.array(range(n+1)).T * h
	u_exact = np.sin(4*np.pi*x) / (16*np.pi*np.pi + 1)
	error[i] = np.sqrt(h) * np.linalg.norm(u_exact - u)
	i += 1
print(error)
logH = np.log(np.array([1./500, 1./200, 1./50, 1./10]))
logError = np.log(error)
# plt.plot(logH, logError)
# plt.show()
p, K = np.linalg.lstsq(np.stack([logH, np.ones(4)]).T, logError.T)[0]
print(p, K)
