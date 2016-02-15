import  numpy as np
import matplotlib.pyplot as plt

def multiply(img, times):
  imgs = []
  for _ in range(times):
      imgs.append(np.clip(img + np.random.normal(scale=0.1, size=img.shape), 0, 1))
# f = plt.figure()
# f.add_subplot(1, 2, 1)
# plt.imshow(1 - img, cmap='Greys')
# f.add_subplot(1, 2, 2)
# plt.imshow(1 - imgs[-1], cmap='Greys')
# plt.show()
  return np.array(imgs)

