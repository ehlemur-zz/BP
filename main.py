import glob
import time

import numpy as np
import pylab as plt

from fastish_pca import fastish_PCA
from kernels import img_kernel, avg_kernel
from multiply import multiply
from representation_learner import learn_representation

from scipy.ndimage import imread
from sklearn.kernel_ridge import KernelRidge

TRAINING_FACES_FILEPATH = 'training.csv'
NEUTRAL_FACES_BLOB = 'faces_dataset/*a*'
HAPPY_FACES_BLOB = 'faces_dataset/*b*'                                     
BOTTLENECK = 1000
IMG_SHAPE = (96, 96)
IMG_MULT = 10

print "Loading faces (training) ..."
start = time.time()
# training_faces = np.loadtxt(TRAINING_FACES_FILEPATH) / 255
# start = time.time() - start
# print "Done in %.2fs" % start
# 
# print "Calculating PCA approx. ..."
# start = time.time()
s, V = learn_representation()
start = time.time() - start
print "Done in %.2fs" % start

print "Found eigenvalues"
print s
plt.plot(s)
plt.show()

print "Loading faces (neutral) ..."

start = time.time()
neutral = []
for filename in sorted(glob.glob(NEUTRAL_FACES_BLOB)):
  neutral += map(lambda x: V.T.dot(x.flatten()), multiply(imread(filename, flatten=True) / 255, IMG_MULT))
neutral = np.array(neutral)
start = time.time() - start
print "Done in %.2fs" % start

print "Loading faces (happy) ..."
start = time.time()
happy = []
for filename in sorted(glob.glob(HAPPY_FACES_BLOB)):
  img = V.T.dot(imread(filename, flatten=True).flatten() / 255)
  happy += [img.copy()  for _ in range(IMG_MULT)]
happy = np.array(happy)
start = time.time() - start
print "Done in %.2fs" % start

print "Performing Ridge Regression ..."
start = time.time()
ridge = KernelRidge(alpha=.5, kernel='linear').fit(neutral, happy)
start = time.time() - start
print "Done in %.2fs" % start

f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(1 - imread('1b_t.jpeg', flatten=True), cmap='Greys')
f.add_subplot(1, 2, 2)
img = imread("test.jpg", flatten=True).flatten() / 255
representation = V.T.dot(img)
represented_prediction = ridge.predict(representation.reshape((1, -1)))
print V.shape, represented_prediction.shape
reconstructed_prediction = V.dot(represented_prediction.flatten()).reshape(IMG_SHAPE)
plt.imshow(1 - reconstructed_prediction, cmap='Greys')
plt.show()

f = plt.figure()
k = 3
for i, filename in enumerate(glob.glob(NEUTRAL_FACES_BLOB)):
  if i >= k:
    break
  filename2 = filename[:-8] + 'b' + filename[-7:] 

  img = imread(filename, flatten=True)
  img2 = imread(filename2, flatten=True)

  representation = V.T.dot(img.flatten())

  reconstruction = V.dot(representation).reshape(IMG_SHAPE)
  prediction = V.dot(ridge.predict(representation.reshape((1, -1))).flatten()).reshape(IMG_SHAPE)
  
  f.add_subplot(k, 4, 4*i + 1)
  plt.imshow(1 - img, cmap='Greys')

  f.add_subplot(k, 4, 4*i + 2)
  plt.imshow(1 - reconstruction, cmap='Greys')
  
  f.add_subplot(k, 4, 4*i + 3)
  plt.imshow(1 - img2, cmap='Greys')

  f.add_subplot(k, 4, 4*i + 4)
  plt.imshow(1 - prediction, cmap='Greys')
plt.show()

