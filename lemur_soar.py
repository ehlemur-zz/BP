import os
import numpy as np

import lemur_util
from lemur_whitening import LemurWhitening

class LemurSoar:
  def __init__(self, neutral, smiling, kernel, repeat=1, alpha=10,
               override=False, filename='SOAR.txt'):

    self.neutral_whiten = LemurWhitening(neutral)
    self.neutral = self.neutral_whiten.whiten(neutral.T)

    self.smiling_whiten = LemurWhitening(smiling)
    self.smiling = self.smiling_whiten.whiten(smiling.T)

    self.alpha = alpha
    self.kernel = kernel
    self.d, self.n = self.neutral.shape

    if os.path.isfile(filename) and not override:
      self.a = np.loadtxt(filename)
      return

    Kx = kernel.x(self.neutral, self.neutral)   # (n, n)
    Ky = kernel.x(self.smiling, self.smiling)    # (n, n)

    self.a = []
    for i in range(self.d):
      if i % 1000 == 0:
        print "feature", i
      Ky_i = Ky / kernel.y(self.smiling, self.smiling, i)  # (n, n)
      M = self.alpha * (Kx + Ky_i) + np.eye(self.n)
      self.a.append(np.linalg.inv(M).dot(self.smiling[i]))

    self.a = np.array(self.a)  # (d, n)
    np.savetxt(filename, self.a)

  def step(self, Y):
    y = np.zeros(Y.shape)
    Ky = self.kernel.x(self.smiling, Y)
    for i in range(self.d):
      Ky_i = (self.Kx + (Ky / self.kernel.y(self.smiling, Y, i)))
      y[i] += self.a[i].dot(Ky_i)

    return y

  def predict(self, X, epochs=300, tol=1e-7):
    X = self.neutral_whiten.whiten(X.T)
    Y = X
    self.Kx = self.alpha * self.kernel.x(self.neutral, X)  # (d, m)
    for _ in range(epochs):
      print "epoch", _
      Y2 = self.step(Y)
      print lemur_util.L2(Y - Y2)
      d = lemur_util.L2(Y - Y2)
      if d < tol:
        break
      Y = Y2
    return self.smiling_whiten.unwhiten(Y).T

