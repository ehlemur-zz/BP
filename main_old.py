# import ..
import glob
import scipy.ndimage

# import .. as ..
import numpy as np
import pylab as plt

# lemur import ..
import lemur_util

# lemur from .. import ..
from lemur_krr import LemurKRR
from lemur_pca import LemurPCA
from lemur_timer import LemurTimer

# LemurPCA parameters
PCA_EPOCHS = 100
PCA_BATCH_SIZE = 13

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

# Load (and repeat) neutral faces
with LemurTimer("loading neutral faces"):
  neutral = []
  for filename in sorted(glob.glob(NEUTRAL_FACES_GLOB)):
    neutral.append(scipy.ndimage.imread(filename, flatten=True).flatten() / 255)
  neutral = np.array(neutral)

# Load (and repeat) smiling faces
with LemurTimer("loading smiling faces"):
  smiling = []
  for filename in sorted(glob.glob(SMILING_FACES_GLOB)):
    smiling.append(scipy.ndimage.imread(filename, flatten=True).flatten() / 255)
  smiling = np.array(smiling)

# Approximate PCA by the lemur method :P
with LemurTimer("approximating PCA"):
  pca = LemurPCA(PCA_BATCH_SIZE, PCA_EPOCHS)

# Perform Kernel Ridge Regression using pca as the kernel
neutral_training = lemur_util.distort(neutral[:N_TRAINING].repeat(REPEAT, axis=0))
smiling_training = smiling[:N_TRAINING].repeat(REPEAT, axis=0)
with LemurTimer("performing KRR"):
  krr = LemurKRR(pca, neutral_training, smiling_training)
 
# Predict the smiling faces from the neutral ones
with LemurTimer("predicting smiling faces"):
  prediction = krr(neutral[-N_TEST:]).T

# Display the neutral, smiling and predicted faces side to side
# one row per test image
f = plt.figure()
for i in range(N_TEST):
  k = N_TRAINING + i

  f.add_subplot(N_TEST, 3, 3*i + 1)
  plt.imshow(1 - neutral[k].reshape(IMG_SHAPE), cmap='Greys')
  f.add_subplot(N_TEST, 3, 3*i + 2)
  plt.imshow(1 - smiling[k].reshape(IMG_SHAPE), cmap='Greys')
  f.add_subplot(N_TEST, 3, 3*i + 3)
  plt.imshow(1 - prediction[i].reshape(IMG_SHAPE), cmap='Greys')

plt.show()

# one row per train image

with LemurTimer("predicting smiling faces"):
  prediction = krr(neutral[:N_TEST]).T

# Display the neutral, smiling and predicted faces side to side
# Predict the smiling faces from the neutral ones
f = plt.figure()
for i in range(N_TEST):
  f.add_subplot(N_TEST, 3, 3*i + 1)
  plt.imshow(1 - neutral[i].reshape(IMG_SHAPE), cmap='Greys')
  f.add_subplot(N_TEST, 3, 3*i + 2)
  plt.imshow(1 - smiling[i].reshape(IMG_SHAPE), cmap='Greys')
  f.add_subplot(N_TEST, 3, 3*i + 3)
  plt.imshow(1 - prediction[i].reshape(IMG_SHAPE), cmap='Greys')

plt.show()
