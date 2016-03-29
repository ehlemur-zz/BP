import numpy as np

import lemur_util

class LemurKernel:
  def __init__(self, d=96):
    self.d = d

  def set_x(self, X1, X2):
    self.X = self._dmatrix(X1, X2)

  def set_y(self, Y1, Y2):
    self.Y1 = Y1
    self.Y2 = Y2
    self.Y = self._dmatrix(Y1, Y2)

  def __call__(self, k):
    K = self._dmatrix(self.Y1[:,k].reshape(-1, 1), self.Y2[:,k].reshape(-1, 1))
    return np.exp(-.5 * (self.X + self.Y - K + 1)/8000)

  def _dmatrix(self, X, Y):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(Y.shape[0]):
      K[:,i] = np.sum((X - Y[i])**2, axis=1)
    return K
    
