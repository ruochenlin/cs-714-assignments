#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

error = np.zeros(4)
i = 0
for n in [999, 499, 99, 49]:
	h = 10. / (n+1)
	data = np.vstack([np.ones(n+2), (-2+h*h)*np.ones(n+2), \
		np.ones(n+2)])
	A = spdiags(data, [-1,0,1], n+2, n+2).toarray()
	A[0][0], A[0][1], A[0][2] = [-1.5*h-h*h, 2*h, -0.5*h]
	A[n+1][n-1], A[n+1][n], A[n+1][n+1] = \
		[0.5*h, -2*h, 1.5*h-h*h]
	f = -h*h*np.exp(np.array(range(n+2)).T * h)
	f[0], f[n+1] = np.zeros(2)
	u = np.linalg.solve(A, f)
	x = np.array(range(n+2)).T * h
	u_exact = -0.5*np.exp(x) + 1./2/np.cos(10)*(np.sin(x) + np.cos(x))
	error[i] = np.sqrt(h)*np.linalg.norm(u_exact - u)
	i += 1
print(error)
# plt.plot(np.log(np.array([10./1000, 10./500, 10./100, 10./50]),\
# 	np.log(error)))
# plt.show()
logH = np.log(np.array([10./1000, 10./500, 10./100, 10./50]))

p, K = np.linalg.lstsq(np.stack([logH, np.ones(4)]).T, np.log(error).T)[0]


print(p, K)
