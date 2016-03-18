import numpy as np

import lemur_util

class LemurKernel:
  def __init__(self, kr=2, kc=3, d=96, s=40):
    self.kr = kr
    self.kc = kc
    self.d = d
    self.s = s

  def x(self, X, Y):
    n = X.shape[1]
    m = Y.shape[1]

    K = np.zeros((n, m))

    for i in range(n):
      x = X.T[i]
      for j in range(m):
        y = Y.T[j]
        d = x - y
        K[i][j] = np.exp(-.5 * (np.dot(d, d))/self.s) 

    return K

  def y(self, X, Y, i):
    n = X.shape[1]
    m = Y.shape[1]

    x = X[i]
    y = Y[i]

    K = np.ones((n, m))
    for i in range(n):
      d = x[i] - y
      K[i] = np.exp(-.5 * d*d/self.s)

    return K
    
