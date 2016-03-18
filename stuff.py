import numpy as np

from scipy.optimize import minimize
from sklearn.linear_model import Lasso

import lemur_util


class LemurIdea:
  def __init__(self, neutral, smiling, batch_size=300, epochs=1,
               alpha=1):
    d = neutral.shape[1]
    self.W = 0
    self.b = 0

    lasso = Lasso(alpha=alpha, warm_start=True)
    lasso.coef_ = np.eye(d, d)

    for _ in range(epochs):
      print "epoch", _
      js = np.random.random_integers(0, neutral.shape[0] - 1, size=batch_size)
      X = lemur_util.distort(neutral[js])
      Y = lemur_util.distort(smiling[js])

      model = lasso.fit(X, Y)
      self.W += model.coef_
      self.b += model.intercept_

    print self.W.shape, self.b.shape

    self.W /= epochs
    self.b /= epochs
    self.b.reshape((1, d))

  def __call__(self, X):
    n = X.shape[0]
    b = np.ones((n, 1)).dot(self.b)
    print self.W.shape, X.T.shape
    return np.dot(self.W, X.T).T + b
