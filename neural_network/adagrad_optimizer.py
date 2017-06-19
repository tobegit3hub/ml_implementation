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

  epoch_number = 1000
  learning_rate = 0.01

  cache = np.array([[0.0], [0.0], [0.0]])
  epsilon = 0.00000001

  for i in range(epoch_number):

    predict = sigmoid(np.dot(features, weights1))
    delta1 = (predict - labels) * predict * (1 - predict)

    grad = np.dot(features.T, delta1)

    cache += grad**2

    weights1 -= learning_rate * grad / (np.sqrt(cache) + epsilon)

    print("Current weights is: {}".format(weights1))

  test_dataset = [[0, 0, 1]]
  predict_propability = sigmoid(np.dot(test_dataset, weights1))
  print("The predict propability is: {}".format(predict_propability))

  test_dataset = [[1, 0, 1]]
  predict_propability = sigmoid(np.dot(test_dataset, weights1))
  print("The predict propability is: {}".format(predict_propability))


if __name__ == "__main__":
  main()
