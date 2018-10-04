#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
import random
import math

error = np.zeros(500)
step = 0.01
for i in range(500):
	H = step * (i + 1)
	for j in range(100):
		h1 = random.uniform(0, H)
		h2 = random.uniform(0, H)
		h3 = random.uniform(0, H)
		a = 2 * (2 * h2 + h3) / h1 / (h1 + h2) \
			/(h1 + h2 + h3)
		c = 2 * (-h1 + h2 + h3) / h2 / h3 / (h1 + h2)
		d = 2 * (h1 - h2) / h3 / (h2 + h3) / \
			(h1 + h2 + h3)
		b = -a - c - d
		error[i] += abs(a*math.exp(1-h1) + \
			b*math.exp(1) + c*math.exp(1+h2) \
			+ d*math.exp(1+h2+h3) - math.exp(1))
	error[i] = error[i] / 100
logError = np.log(error)
logH = np.log(np.array(range(1, 501)) * step)
# plt.scatter(logH, logError)
# plt.xlabel('log H')
# plt.ylabel('log Error')
# plt.show()
A = np.vstack([logH, np.ones(500)]).T
p, K = np.linalg.lstsq(A, logError)[0]
print(p, K)
