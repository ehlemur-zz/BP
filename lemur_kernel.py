import numpy as np

import lemur_util

class LemurKernel:
  def __init__(self, d=96, s=500):
    self.d = d
    self.s = s

  def set_x(self, X1, X2):
    self.X = X1.dot(X2.T); #np.exp(-.5 * self._dmatrix(X1, X2) / self.s)

  def set_y(self, Y1, Y2):
    self.Y1 = Y1
    self.Y2 = Y2
    self.Y = np.dot(Y1, Y2.T)

  def __call__(self, k):
    K = np.dot(self.Y1[:,k].reshape(-1, 1), self.Y2[:,k].reshape(-1, 1).T)
    return self.X * (self.Y - K)

  def _dmatrix(self, X, Y):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(Y.shape[0]):
      K[:,i] = np.sum((X - Y[i])**2, axis=1)
    return K

    
