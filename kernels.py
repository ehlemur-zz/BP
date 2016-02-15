import numpy as np

def img_kernel(kernel, img_shape, levels=3, split=2):
  def k(img1, img2):
    img1 = img1.reshape(img_shape)
    img2 = img2.reshape(img_shape)
    answer = 0
    for l in range(levels):
      n = split ** (levels - l - 1)
      a = split ** l
      hs = img_shape[0] / n
      ws = img_shape[1] / n
      if hs == 0 or ws == 0:
        print l, n, hs, ws
      for hi in range(n):
        fr_h = hs * hi
        to_h = hs * (hi + 1) if hi < n - 1 else img_shape[0] - 1
        for wi in range(n):
          fr_w = ws * wi
          to_w = ws * (wi + 1) if wi < n - 1 else img_shape[0] - 1
          answer += a * kernel(img1[fr_h:to_h,fr_w:to_w].flatten(), 
                               img2[fr_h:to_h,fr_w:to_w].flatten())
    img1.flatten()
    img2.flatten()
    return answer
  return k

def avg_kernel(sigma=2):
  def k(x, y):
    return np.exp(-2*(np.abs(np.mean(x) - np.mean(y))) / sigma)
  return k
