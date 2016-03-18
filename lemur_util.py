import numpy as np

def distort(A, scale=.01):
  return np.clip(A + np.random.normal(scale=scale, size=A.shape), 0, 1)

def dot(*Ms):
  n = len(Ms)
  shapes = map(lambda M: M.shape[0], Ms) + [Ms[-1].shape[1]]

  best = np.inf * np.ones((n, n))
  best_cut = np.zeros((n, n), int)

  for fr in range(n):
    best[fr][0] = 0

  for span in range(1, n):
    for fr in range(n - span):
      to = fr + span  # [fr, to)
      for cut in range(span):
        mi = fr + cut + 1  # [fr, mi) u [mi, to)
        candidate = best[fr][cut] + best[mi][span-cut-1] + shapes[fr] * shapes[mi] * shapes[to]
        if best[fr][span] > candidate:
          best[fr][span] = candidate
          best_cut[fr][span] = cut
  
  def go(fr, span):
    if span == 0:
      return Ms[fr]
    cut = best_cut[fr][span]
    mi = fr + cut + 1
    return go(fr, cut).dot(go(mi, span-cut-1))

  return go(0, n-1)

def L2(x):
  return np.sqrt(np.sum(x**2))

def L2_squared(x):
  return np.sum(x**2)
