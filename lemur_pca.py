import os.path
import threading

import numpy as np

import lemur_util

from lemur_timer import LemurTimer

class LemurPCA:
  def __init__(self, batch_size, n_epochs=None, faces_csv='faces.csv',
               output_filename='lemur_pca.txt', override=False):
    if os.path.isfile(output_filename) and not override:
      self.H = np.loadtxt(output_filename)
      return
    
    with LemurTimer("loading input file..."):
      self.A = np.loadtxt(faces_csv) / 255

    self.batch_size = batch_size

    self.pca = threading.Lock()
    self.output_file = threading.Lock()
    self.output_filename = output_filename

    d = self.A.shape[1]
    self.H = np.zeros((d, d))

    try:
      if n_epochs is not None:
        for _ in range(n_epochs):
          self._update()
      else:
        self._update()
        while True:
          update_thread = _UpdateThread(self)
          save_thread = _SaveThread(self) 
          update_thread.start()
          save_thread.start()
          update_thread.join()
    finally:
      self._save()

  def __call__(self, X, Y):
    return X.dot(self.H.dot(Y.T))

  def _update(self):
    n = self.A.shape[0]
    js = np.random.random_integers(0, n - 1, size=self.batch_size)
    batch = lemur_util.distort(self.A[js])
    self.pca.acquire()
    self.H += batch.T.dot(batch)
    self.pca.release()
  
  def _save(self):
    if self.output_file.acquire(False):
      with LemurTimer("saving file"):
        self.pca.acquire()
        A = self.H.copy()
        self.pca.release()
        np.savetxt(self.output_filename, A)
        self.output_file.release()

class _UpdateThread(threading.Thread):
  def __init__(self, pca):
    threading.Thread.__init__(self)
    self.pca = pca
  def run(self):
    with LemurTimer("epoch"):
      self.pca._update()

class _SaveThread(threading.Thread):
  def __init__(self, pca):
    threading.Thread.__init__(self)
    self.pca = pca
  def run(self):
    self.pca._save()

