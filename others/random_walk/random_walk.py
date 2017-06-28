#!/usr/bin/env python

import os
import random
import matplotlib
import matplotlib.pyplot as plt

def main():
  value = 100
  values = [value]
  steps = 10000

  for i in xrange(steps):
    if random.randint(0, 1) == 1:
      choose = 1
    else:
      choose = -1
    value += choose
    values.append(value)

  plt.plot(range(len(values)), values)
  plt.show()

if __name__ == "__main__":
  main()
