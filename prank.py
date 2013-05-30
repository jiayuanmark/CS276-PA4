import numpy as np
from itertools import izip
from utilFunction import vector_sum, vector_scale, vector_diff

class PRank:
  def __init__(self):
    pass

  def fit(self, X, Y):
    self.rel = [-1.0, 0.0, 1.0, 2.0, 3.0]
    self.w = [0.0] * len(X[1])
    self.k = len(self.rel)
    k = self.k
    self.b = [0.0] * (k - 1)
    yr = [0.0] * (k - 1)
    tr = [0.0] * (k - 1)
    
    for x, y_val in izip(X, Y):
      new_y = self.predict_point(x)
      y = 0
      for i, rel_val in enumerate(self.rel):
        if abs(y_val - rel_val) < abs(y_val - self.rel[y]):
          y = i
  
      if new_y != y:
        for r in range(k - 1):
          yr[r] = -1 if y <= r else 1
        for r in range(k - 1):
          tr[r] = yr[r] if yr[r] * (np.dot(x, self.w) - self.b[r]) <= 0 else 0
        self.w = vector_sum(self.w, vector_scale(x, sum(tr)))
        self.b = vector_diff(self.b, tr)
  
  def predict_point(self, x):
    w = np.dot(x, self.w)
    for r in range(self.k - 1):
      if w < self.b[r]:
        return r
    return self.k - 1
  
  def predict(self, X):
    return [self.rel[self.predict_point(x)] for x in X]
    