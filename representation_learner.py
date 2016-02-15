import os.path
import numpy as np

from fastish_pca import fastish_PCA

def learn_representation(*args, **kwargs):
  basename = kwargs.get('basename', 'PCA')
  override = kwargs.get('override', False)

  V_file = basename + '_V.txt'
  s_file = basename + '_s.txt'

  if os.path.isfile(V_file) and not override:
    return np.loadtxt(s_file), np.loadtxt(V_file)

  _, s, V = fastish_PCA(*args)
  np.savetxt(V_file, V)
  np.savetxt(s_file, s)
  return s, V 


