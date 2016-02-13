import numpy as np

def fastish_PCA(A, k, i, power_method=False):
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
  Ut = Q.dot(W.T)

  if transposed:
    return Vt[:,:k], s[:k], Ut[:,:k]
  else:
    return Ut[:,:k], s[:k], Vt[:,:k]
