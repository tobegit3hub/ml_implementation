#!/usr/bin/env python

import numpy as np
import redis


def sigmoid(x):
  return 1.0 / (1 + np.exp(-x))


def test_sigmoid():
  x = 0
  print("Input x: {}, the sigmoid value is: {}".format(x, sigmoid(x)))


def main():

  client = redis.StrictRedis(host='localhost', port=6379, db=0)

  # Prepare dataset
  train_features = np.array([[1, 0, 26], [0, 1, 25]], dtype=np.float)
  train_labels = np.array([1, 0], dtype=np.int)

  test_features = np.array([[1, 0, 26], [0, 1, 25]], dtype=np.float)
  test_labels = np.array([1, 0], dtype=np.int)

  feature_size = 3
  batch_size = 2

  # Define hyperparameters
  epoch_number = 10
  learning_rate = 0.01

  # Start training
  for epoch_index in range(epoch_number):
    print("Start the epoch: {}".format(epoch_index))

    # Train with single example
    train_features = np.array([1, 0, 25], dtype=np.float)
    train_labels = np.array([0], dtype=np.int)

    if client.exists("weights0") and client.exists(
        "weights1") and client.exists("weights2"):
      weights0 = float(client.get("weights0"))
      weights1 = float(client.get("weights1"))
      weights2 = float(client.get("weights2"))
    else:
      weights0 = 1.0
      weights1 = 1.0
      weights2 = 1.0

    weights = np.array([weights0, weights1, weights2])

    # [3] = [3] * [3]
    multiple_weights_result = train_features * weights
    # [1] = [3]
    predict = np.sum(multiple_weights_result)
    # [1] = [1]
    sigmoid_predict = sigmoid(predict)
    # [1] = [1]
    predict_difference = train_labels - sigmoid_predict
    # [3] = [3] * [1]
    grad = train_features * predict_difference
    # [3] = [3]
    weights += learning_rate * grad
    print("Current weights is: {}".format(weights))

    client.set("weights0", weights[0])
    client.set("weights1", weights[1])
    client.set("weights2", weights[2])

    # TODO: Predict with validate dataset
    predict_true_probability = sigmoid(np.sum(train_features * weights))
    print("Current predict true probability is: {}".format(
        predict_true_probability))

    likehood = 1 - predict_true_probability
    print("Current likehood is: {}\n".format(likehood))


if __name__ == "__main__":
  main()
