#!/usr/bin/env python

import numpy as np


def compute_mutual_information(x, y):
  # Example of x: [2, 2, 3, 2]
  # Example of y: [1, 2, 4, 2]
  result = 0.0

  # Example: [2, 3]
  x_values = np.unique(x)

  # Example: [1, 2, 4]
  y_values = np.unique(y)

  # Example: [0.75, 0.25]
  x_propabilities = np.array(
      [len(x[x == xval]) / float(len(x)) for xval in x_values])

  # Example: [0.25, 0.5, 0.25]
  y_propabilities = np.array(
      [len(y[y == yval]) / float(len(y)) for yval in y_values])

  for current_index in xrange(len(x_values)):
    if x_propabilities[current_index] == 0.:
      continue

    current_value = x_values[current_index]
    # Example: [True, True, False, True]
    selected_indexs_with_same_current_value = [x == current_value]

    # Example: [1, 2, 2]
    selected_y_values_with_last_indexs = y[x == current_value]
    if len(selected_y_values_with_last_indexs) == 0:
      continue
    sy = selected_y_values_with_last_indexs

    # Example: [0.25, 0.5, 0] with the sample shape of y_propabilities
    xy_propabilities = np.array(
        [len(sy[sy == yval]) / float(len(y)) for yval in y_values])

    pxy = xy_propabilities  # P(x, y)
    px = x_propabilities[current_index]  # P(x)
    py = y_propabilities  # P(y)

    # Example: [3, 0]
    temp = pxy[py > 0.] / py[py > 0.] / px  # (P(x,y) / (P(x) * P(y))
    #result += sum(pxy[temp>0]*np.log2( temp[temp>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    result += sum(
        pxy[temp > 0] *
        np.log(temp[temp > 0]))  # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )

  return result


def main():
  x = np.array([2, 2, 3, 2])
  y = np.array([1, 2, 4, 2])

  result = compute_mutual_information(x, y)
  print(result)


if __name__ == "__main__":
  main()
