from lemur_pca import LemurPCA
from lemur_timer import LemurTimer

with LemurTimer("Approximating PCA ..."):
  BATCH_SIZE = 100
  LemurPCA(BATCH_SIZE, override=True)
