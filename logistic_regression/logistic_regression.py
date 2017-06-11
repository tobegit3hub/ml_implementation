#!/usr/bin/env python

import numpy as np


def sigmoid(x):
  return 1.0 / (1 + np.exp(-x))


def test_sigmoid():
  x = 0
  print("Input x: {}, the sigmoid value is: {}".format(x, sigmoid(x)))


def main():
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
  weights = np.ones(feature_size)

  # Start training
  for epoch_index in range(epoch_number):
    print("Start the epoch: {}".format(epoch_index))
    ''' Implement with batch size
    
    # [2, 3] = [3] * [2, 3]
    multiple_weights_result = weights * train_features
    # [2] = [2, 3]
    predict = np.sum(multiple_weights_result, 1)
    # [2] = [2]
    sigmoid_predict = sigmoid(predict)
    # [2] = [2]
    predict_difference = train_labels - sigmoid_predict
    # TODO: [2, 3, 1] = [2, 3] * [2]
    batch_grad = train_features * predict_difference
    # TODO: fix that
    grad = batch_grad
    # [3, 1] = [3, 1]
    weights += learning_rate * grad
    '''

    # Train with single example
    train_features = np.array([1, 0, 25], dtype=np.float)
    train_labels = np.array([0], dtype=np.int)

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

    # TODO: Predict with validate dataset
    predict_true_probability = sigmoid(np.sum(train_features * weights))
    print("Current predict true probability is: {}".format(
        predict_true_probability))

    likehood = 1 - predict_true_probability
    print("Current likehood is: {}\n".format(likehood))


if __name__ == "__main__":
  main()
