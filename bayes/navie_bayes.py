#!/usr/bin/env python

import math
import numpy as np


def mean(numbers):
  return 1.0 * sum(numbers) / len(numbers)


def test_mean():
  numbers = [1, 2, 3, 4, 5]
  result = mean(numbers)
  print("Numbers are: {} and mean is: {}".format(numbers, result))


def stdev(numbers):
  mean_value = mean(numbers)

  sum_difference_square = 0.0
  for x in numbers:
    difference = x - mean_value
    difference_square = pow(difference, 2)
    sum_difference_square += difference_square

  # TODO: Remove "-1" or not
  # variance = 1.0 * sum_difference_square / (len(numbers) - 1)
  variance = 1.0 * sum_difference_square / len(numbers)
  stdev = math.sqrt(variance)
  return stdev


def test_stdev():
  numbers = [1, 2, 3, 4, 5]
  result = stdev(numbers)
  print("Numbers are: {} and stdev is: {}".format(numbers, result))


def seperate_by_label(dataset):
  # Example: [[6,148,72,35,0,33.6,0.627,50,1], [1,85,66,29,0,26.6,0.351,31,0]]

  # Example: {0: [[2, 21, 0]], 1: [[1, 20, 1], [3, 22, 1]]}
  label_instances_map = {}

  for i in range(len(dataset)):
    instance = dataset[i]
    label = instance[-1]

    if not label_instances_map.has_key(label):
      label_instances_map[label] = []

    label_instances_map[label].append(instance)

  return label_instances_map


def test_seperate_by_label():
  dataset = [[1, 20, 1], [2, 21, 0], [3, 22, 1]]

  # Should be {0: [[2, 21, 0]], 1: [[1, 20, 1], [3, 22, 1]]}
  label_instances_map = seperate_by_label(dataset)
  print("Dataset is: {} and seperate by label result is: {}".format(
      dataset, label_instances_map))


def get_mean_and_stdev(dataset):
  # [[1, 20, 0], [2, 21, 1], [3, 22, 0]] -> [(1, 2, 3), (20, 21, 22), (0, 1, 0)] -> [(2.0, 1.0), (21.0, 1.0)]
  mean_and_stdev_list = [(mean(feature), stdev(feature))
                         for feature in zip(*dataset)]
  mean_and_stdev_list_without_label = mean_and_stdev_list[:-1]

  return mean_and_stdev_list_without_label


def test_get_mean_and_stdev():
  dataset = [[1, 20, 0], [2, 21, 1], [3, 22, 0]]

  # Should be [(2.0, 1.0), (21.0, 1.0)]
  result = get_mean_and_stdev(dataset)

  print("Dataset is: {} and result of get_mean_and_stdev is: {}").format(
      dataset, result)


def get_mean_and_stdev_by_label(dataset):
  label_instances_map = seperate_by_label(dataset)

  # Example: {0: [(2.0, 1.0), (21.0, 1.0)], 1: [(2.0, 0.0), (21.0, 0.0)]}
  label_meanstdevlist_map = {}

  for label, instances in label_instances_map.iteritems():
    label_meanstdevlist_map[label] = get_mean_and_stdev(instances)

  return label_meanstdevlist_map


def test_get_mean_and_stdev_by_label():
  dataset = [[1, 20, 0], [2, 21, 1], [3, 22, 0]]

  # Should be {0: [(2.0, 1.0), (21.0, 1.0)], 1: [(2.0, 0.0), (21.0, 0.0)]}
  result = get_mean_and_stdev_by_label(dataset)
  print("Dataset is: {} and the result of label_meanstdevlist_map is: {}"
        ).format(dataset, result)


def calculate_gauss_probability(x, mean, stdev):
  # exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
  # return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

  # TODO: Handle when stdev is 0.0 for small dataset
  left_coefficient = 1.0 / (math.sqrt(2 * math.pi) * stdev)
  right_exponent = -1.0 * math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))
  result = left_coefficient * math.exp(right_exponent)
  return result


def test_calculate_gauss_probabiity():
  x = 71.5
  mean = 73
  stdev = 6.2

  # Should be 0.0624896575937
  probability = calculate_gauss_probability(x, mean, stdev)
  print("The probability is: {}".format(probability))


def calculate_gauss_probabilities_by_label(label_meanstdevlist_map, instance):
  label_probability_map = {}

  # Example: {0: [(2.0, 1.0), (21.0, 1.0)], 1: [(2.0, 0.0), (21.0, 0.0)]}
  for label, mean_stdev_list in label_meanstdevlist_map.iteritems():
    label_probability_map[label] = 1

    for i in range(len(mean_stdev_list)):
      mean, stdev = mean_stdev_list[i]
      x = instance[i]
      label_probability_map[label] *= calculate_gauss_probability(
          x, mean, stdev)

  return label_probability_map


def test_calculate_gauss_probabilities_by_label():
  label_meanstdevlist_map = {0: [(1, 0.5)], 1: [(20, 5.0)]}
  instance = [1.1, '?']
  label_probability_map = calculate_gauss_probabilities_by_label(
      label_meanstdevlist_map, instance)

  # Probabilities for each class: {0: 0.7820853879509118, 1: 6.298736258150442e-05}
  print("The label_probability_map is: {}".format(label_probability_map))


def predict(label_meanstdevlist_map, instance):

  label_probability_map = calculate_gauss_probabilities_by_label(
      label_meanstdevlist_map, instance)

  best_propability = 0
  best_label = None

  for label, probability in label_probability_map.iteritems():

    if probability > best_propability:
      best_propability = probability
      best_label = label

  return best_label


def main():

  # [5, 9]
  dataset = [[6, 148, 72, 35, 0, 33.6, 0.627, 50,
              1], [1, 85, 66, 29, 0, 26.6, 0.351, 31,
                   0], [8, 183, 64, 0, 0, 23.3, 0.672, 32,
                        1], [2, 89, 68, 23, 94, 28.1, 0.167, 21, 0],
             [0, 137, 40, 35, 168, 43.1, 2.288, 33, 1]]

  # [8]
  test_dataset = [7, 147, 72, 35, 0, 33.6, 0.628, 50]

  label_meanstdevlist_map = get_mean_and_stdev_by_label(dataset)

  result = predict(label_meanstdevlist_map, test_dataset)

  print("Test dataset is: {} and result is: {}".format(test_dataset, result))


if __name__ == "__main__":
  main()
