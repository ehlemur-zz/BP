# import ..
import glob
import scipy.ndimage

# import .. as ..
import numpy as np
import pylab as plt

# lemur import ..
import lemur_util

# lemur from .. import ..
from lemur_kernel import LemurKernel
from lemur_pca import lemur_PCA
from lemur_soar import LemurSOAR
from lemur_timer import LemurTimer

# A bunch of faces
FACES = 'faces.txt'
BOTTLENECK = 3000

# Neutral and smiling faces
NEUTRAL_FACES_GLOB = 'faces/*a*'
SMILING_FACES_GLOB = 'faces/*b*'

# Image information
N_IMAGES = 200
IMG_SHAPE = (96, 96)
REPEAT = 1

# Testing and training size
N_TEST = 5
N_TRAINING = N_IMAGES - N_TEST

# Load neutral faces
with LemurTimer("loading neutral faces"):
  neutral = []
  for filename in sorted(glob.glob(NEUTRAL_FACES_GLOB)):
    neutral.append(scipy.ndimage.imread(filename, flatten=True).flatten() / 255)
  neutral = np.array(neutral)

# Load smiling faces
with LemurTimer("loading smiling faces"):
  smiling = []
  for filename in sorted(glob.glob(SMILING_FACES_GLOB)):
    smiling.append(scipy.ndimage.imread(filename, flatten=True).flatten() / 255)
  smiling = np.array(smiling)

# Shuffle images
p = np.random.permutation(smiling.shape[0])

neutral = neutral[p]
smiling = smiling[p]

with LemurTimer("calculating PCA"):
  V = lemur_PCA(FACES, BOTTLENECK, 2)

  neutral_pca = neutral.dot(V)
  smiling_pca = smiling.dot(V)

with LemurTimer("training lemur_idea"):
  neutral_training = np.append(neutral_pca[:N_TRAINING],
                               smiling_pca[:N_TRAINING], axis=0)
  smiling_training = np.append(smiling_pca[:N_TRAINING],
                               smiling_pca[:N_TRAINING], axis=0)
  lemur_soar = LemurSOAR(neutral_training, smiling_training,
                         LemurKernel(), override=True)

with LemurTimer("predicting smiling faces from test set"):
  prediction = lemur_soar.predict(neutral_pca[N_TRAINING:]).dot(V.T)

# Display the neutral, smiling and predicted faces side to side
# one row per test image
f = plt.figure()
for i in range(N_TEST):
  k = N_TRAINING + i

  f.add_subplot(N_TEST, 4, 4*i + 1)
  plt.imshow(1 - neutral[k].reshape(IMG_SHAPE), cmap='Greys')
  f.add_subplot(N_TEST, 4, 4*i + 2)
  plt.imshow(1 - smiling[k].reshape(IMG_SHAPE), cmap='Greys')
  f.add_subplot(N_TEST, 4, 4*i + 3)
  plt.imshow(1 - neutral_pca[k].dot(V.T).reshape(IMG_SHAPE), cmap='Greys')
  f.add_subplot(N_TEST, 4, 4*i + 4)
  plt.imshow(1 - prediction[i].reshape(IMG_SHAPE), cmap='Greys')

plt.show()


with LemurTimer("predicting smiling faces from test set"):
  prediction = lemur_soar.predict(smiling_pca[N_TRAINING:]).dot(V.T)

# Display the neutral, smiling and predicted faces side to side
# one row per test image
f = plt.figure()
for i in range(N_TEST):
  k = N_TRAINING + i

  f.add_subplot(N_TEST, 4, 4*i + 1)
  plt.imshow(1 - neutral[k].reshape(IMG_SHAPE), cmap='Greys')
  f.add_subplot(N_TEST, 4, 4*i + 2)
  plt.imshow(1 - smiling[k].reshape(IMG_SHAPE), cmap='Greys')
  f.add_subplot(N_TEST, 4, 4*i + 3)
  plt.imshow(1 - neutral_pca[k].dot(V.T).reshape(IMG_SHAPE), cmap='Greys')
  f.add_subplot(N_TEST, 4, 4*i + 4)
  plt.imshow(1 - prediction[i].reshape(IMG_SHAPE), cmap='Greys')

plt.show()

# one row per train image

with LemurTimer("predicting smiling faces from training set"):
  prediction = lemur_soar.predict(neutral_pca[:N_TEST]).dot(V.T)

# Display the neutral, smiling and predicted faces side to side
# Predict the smiling faces from the neutral ones
f = plt.figure()
for i in range(N_TEST):
  f.add_subplot(N_TEST, 4, 4*i + 1)
  plt.imshow(1 - neutral[i].reshape(IMG_SHAPE), cmap='Greys')
  f.add_subplot(N_TEST, 4, 4*i + 2)
  plt.imshow(1 - smiling[i].reshape(IMG_SHAPE), cmap='Greys')
  f.add_subplot(N_TEST, 4, 4*i + 3)
  plt.imshow(1 - neutral_pca[i].dot(V.T).reshape(IMG_SHAPE), cmap='Greys')
  f.add_subplot(N_TEST, 4, 4*i + 4)
  plt.imshow(1 - prediction[i].reshape(IMG_SHAPE), cmap='Greys')

plt.show()
