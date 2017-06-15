#!/usr/bin/env python


def main():
  print("Start training")

  # [4, 5]
  dataset = [[2.0, 3.0, 0.0, 5.0, 2.8], [3.0, 4.0, 5.0, 0.0, 0.0],
             [0.5, 0.3, 0.7, 0.9, 0.0], [-2.0, 0.1, 0.2, 0.9, 0.0]]
  # [4]
  labels = [1, 1, -1, -1]

  feature_size = len(dataset[0])
  instance_number = len(dataset)

  epoch_number = 100
  learning_rate = 0.1
  lambda_rate = 0.01

  # TODO: [6] which contains bias
  weights = [1.0, 1.0, 1.0, 1.0, 1.0]

  for epoch_index in range(epoch_number):

    for i in range(instance_number):

      wx = 0.0
      instance = dataset[i]
      label = labels[i]

      grad = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

      for j in range(feature_size):
        wx += weights[j] * instance[j]

      if 1 - wx * label > 0:
        for j in range(feature_size):
          grad[j] = lambda_rate * weights[j] - instance[j] * label
      else:
        grad[j] = lambda_rate * weights[j]

      for j in range(feature_size):
        weights[j] -= learning_rate * grad[j]
      print("Current weights: {}".format(weights))

  test_instance = [2.0, 3.0, 0.0, 5.0, 2.8]
  predict(weights, test_instance)

  test_instance = [0.5, 0.3, 0.7, 0.9, 0.0]
  predict(weights, test_instance)


def predict(weights, instance):
  # Example: [2.0, 3.0, 0.0, 5.0, 2.8]

  feature_size = len(instance)
  wx = 0.0

  for i in range(feature_size):
    wx += weights[i] * instance[i]

  if wx > 0:
    predict_value = 1
  else:
    predict_value = -1

  print(
      "Predict instance: {} and result is: {}".format(instance, predict_value))
  return predict_value


if __name__ == "__main__":
  main()
