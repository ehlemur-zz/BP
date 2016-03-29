import os.path
import numpy as np

import lemur_util

def lemur_PCA(in_filename, k, i, power_method=False, out_filename='PCA.txt',
              override=False):
  if not override and os.path.isfile(out_filename):
    return np.loadtxt(out_filename)

  A = np.loadtxt(in_filename) / 255

  transposed = False
  if A.shape[0] < A.shape[1]:
    A = A.T
    transposed = True

  m = A.shape[0]
  n = A.shape[1]
  l = k + 2

  G = np.random.randn(n, l)

  if power_method:
    H = A.dot(G)
    for _ in range(i):
      H = A.dot(A.T.dot(H))
  else:
    H = [A.dot(G)]
    for _ in range(i):
      H.append(A.dot(A.T.dot(H[-1])))
    H = np.array(map(np.transpose, H)).reshape((i + 1) * l, m).T

  Q = np.linalg.qr(H)[0]
  T = A.T.dot(Q)
  Vt, s, W = np.linalg.svd(T, full_matrices=False)

  if transposed:
    V = Q.dot(W.T)[:,:k]
  else:
    V = Vt[:,:k]

  np.savetxt(out_filename, V)
  return V
