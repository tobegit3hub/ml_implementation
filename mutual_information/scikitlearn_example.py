#!/usr/bin/env python

import numpy as np
from sklearn.metrics import mutual_info_score

x = np.array([2, 2, 3, 2])
y = np.array([1, 2, 4, 2])
result = mutual_info_score(x, y)

print(result)
