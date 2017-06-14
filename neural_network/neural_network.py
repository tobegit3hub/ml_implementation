#!/usr/bin/env python

import math
import numpy as np


def sigmoid(x):
  return 1.0 / (1 + np.exp(-x))


def sigmoid_derivate(x):
  return x * (1 - x)


def main():
  np.random.seed(1)

  # [4, 3]
  features = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

  # [4, 1]
  labels = np.array([[0], [0], [1], [1]])

  # weights1 = 2 * np.random.random((3,1)) - 1
  weights1 = np.array([[1.0], [1.0], [1.0]])

  epoch_number = 100
  learning_rate = 0.01

  for i in range(epoch_number):
    input_layer = features
    layer1 = sigmoid(np.dot(input_layer, weights1))
    difference1 = labels - layer1
    delta1 = -1.0 * difference1 * sigmoid_derivate(layer1)

    grad = np.dot(input_layer.T, delta1)
    weights1 -= learning_rate * grad

    print("Current weights is: {}".format(weights1))

  test_dataset = [[0, 0, 1]]
  predict_propability = sigmoid(np.dot(test_dataset, weights1))
  print("The predict propability is: {}".format(predict_propability))


if __name__ == "__main__":
  main()
