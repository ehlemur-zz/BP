import numpy as np

class LemurWhitening:
  def __init__(self, imgs):
    self.m = np.mean(imgs, axis=0)
    imgs -= self.m
    self.s = np.std(imgs, axis=0) + 1e-6
  def whiten(self, X):
    return X
    return ((X.T - self.m) / self.s).T
  def unwhiten(self, X):
    return X
    return (X.T * self.s + self.m).T

