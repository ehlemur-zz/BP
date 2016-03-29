import os
import numpy as np

import sklearn.linear_model

import lemur_util
from lemur_whitening import LemurWhitening

class LemurSoar:
  def __init__(self, neutral, smiling, kernel, repeat=1, alpha=1,
               override=False, filename='SOAR.txt'):
    self.n, self.d = smiling.shape

    M = np.linalg.inv(neutral.dot(neutral.T) + alpha * np.eye(self.n))
    self.W = lemur_util.dot(neutral.T, M, smiling)
    
    self.neutral = neutral
    self.smiling = smiling

    self.kernel = kernel

    if os.path.isfile(filename) and not override:
      self.a = np.loadtxt(filename)
      return

    kernel.set_x(self.neutral, self.neutral)   # (n, n)
    kernel.set_y(self.smiling, self.smiling)   # (n, n)

    self.a = []
    for i in range(self.d):
      if i % 1000 == 0:
        print "feature", i
      self.a.append(np.linalg.inv(kernel(i) + alpha * np.eye(self.n))
                    .dot(smiling[:,i]))
    
    self.a = np.array(self.a)
    np.savetxt(filename, self.a)

  def step(self, Y):
    y = np.zeros(Y.shape)
    self.kernel.set_y(Y, self.smiling)
    for i in range(self.d):
      y[:,i] = self.kernel(i).dot(self.a[i])
    return y

  def predict(self, X, epochs=300, tol=1e-7):
    Y = X.dot(self.W)
    self.kernel.set_x(X, self.neutral)
    for _ in range(epochs):
      Y2 = self.step(Y)
      d = lemur_util.L2(Y - Y2)
      if d < tol:
        break
      Y = Y2
    return Y
  
  def score(self, X, Y):
    return lemur_util.L2_squared(self.predict(X) - Y) / X.shape[0]

