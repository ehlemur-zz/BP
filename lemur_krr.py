import numpy as np

class LemurKRR:
  def __init__(self, kernel, X, Y, alpha=.1):
    n = X.shape[0]
    self.k = kernel
    self.X = X
    K = self.k(self.X, self.X)
    self.A = np.dot(Y.T, np.linalg.inv(K + alpha * np.identity(n)))

  def _k(self, x):
    return self.k(self.X, x)

  def __call__(self, x):
    return self.A.dot(self._k(x))
