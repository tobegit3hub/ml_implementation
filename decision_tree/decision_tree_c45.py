#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import operator

def calculateShannonEntropy(dataset):
    # [[0, 0, 0, 0, 'N'], [0, 0, 1, 1, 'Y']]
    instance_number = len(dataset)

    # {'Y': 1, 'N': 1}
    label_number_map = {}

    for instance in dataset:
        label = instance[-1]
        if label not in label_number_map.keys():
            label_number_map[label] = 0
        label_number_map[label] += 1

    total_shannon_entropy = 0.0
    for label in label_number_map:
        probability = float(label_number_map[label]) / instance_number
        shannon_entropy = probability * math.log(probability, 2) * -1
        total_shannon_entropy += shannon_entropy

    return total_shannon_entropy


def testCalculateShannonEntropy():
  # Should be 1.0
  dataset = [[0, 0, 0, 0, 'N'], [0, 0, 1, 1, 'Y']]
  print("The shannon entropy is: {}".format(calculateShannonEntropy(dataset)))

  # Should be 0.0
  dataset = [[0, 0, 0, 0, 'N'], [0, 0, 1, 1, 'N']]
  print("The shannon entropy is: {}".format(calculateShannonEntropy(dataset)))


def split_dataset(dataset, feature, value):
    """ Get the dataset when "feature" is equal to "value"
    """
    reture_data_set = []

    # TODO: Example
    for instance in dataset:
        if instance[feature] == value:
            new_instance = instance[:feature]
            new_instance.extend(instance[feature+1:])
            reture_data_set.append(new_instance)
    return reture_data_set


def choose_best_feature_to_split(dataset):
    # Example: 4
    feature_number = len(dataset[0]) - 1
    base_entropy = calculateShannonEntropy(dataset)

    best_info_gain_ratio = 0.0
    best_feature = -1

    # Example: [0, 0, 0, 0]
    for i in range(feature_number):
        # Example:
        instance_with_one_feature = [instance[i] for instance in dataset]
        feature_value_set = set(instance_with_one_feature)
        after_split_entropy = 0.0
        instrinsic_value = 0.0

        # Example: [0, 1]
        for value in feature_value_set:
            sub_dataset = split_dataset(dataset, i, value)

            probability = len(sub_dataset) / float(len(dataset))
            after_split_entropy += probability * calculateShannonEntropy(sub_dataset)
            instrinsic_value += -probability * math.log(probability, 2)

        info_gain = base_entropy - after_split_entropy

        # Check if it is zero
        if (instrinsic_value == 0):
            continue

        info_gain_ratio = info_gain / instrinsic_value

        if (info_gain_ratio > best_info_gain_ratio):
            best_info_gain_ratio = info_gain_ratio
            best_feature = i

    return best_feature






def create_decision_tree(dataset, header_names):

    # Example: [[0, 0, 0, 0, 'N'], [0, 0, 0, 1, 'N'], [1, 0, 0, 0, 'Y']]

    # Example: ['N', 'N', 'Y']
    labels = [instance[-1] for instance in dataset]

    if labels.count(labels[0]) == len(labels):
        # Return if all the values are the same
        return labels[0]

    # Example: ['N']
    if len(dataset[0]) == 1:

        label_count_map = {}
        for label in labels:
            if label not in label_count_map.keys():
                label_count_map[label] = 0
            label_count_map[label] += 1
        sorted_label_count_map = sorted(label_count_map.iteritems(), key=operator.itemgetter(1), reversed=True)
        return sorted_label_count_map[0][0]


    best_feature_id = choose_best_feature_to_split(dataset)

    header_name = header_names[best_feature_id]
    decision_tree = {header_name: {}}
    # TODO: don't modify the input parameter
    del(header_names[best_feature_id])

    all_feature_values = [instance[best_feature_id] for instance in dataset]
    unique_feature_values = set(all_feature_values)

    for value in unique_feature_values:
        sub_header_names = header_names[:]
        sub_dataset = split_dataset(dataset, best_feature_id, value)
        decision_tree[header_name][value] = create_decision_tree(sub_dataset, sub_header_names)

    return decision_tree

def predict(decision_tree, header_names, test_dataset):

    # Example: {'outlook': {0: 'N', 1: 'Y', 2: {'windy': {0: 'Y', 1: 'N'}}}}
    # print("Current tree: {}".format(decision_tree))

    # Example: "outlook"
    root_key = list(decision_tree.keys())[0]

    # Example: {0: 'N', 1: 'Y', 2: {'windy': {0: 'Y', 1: 'N'}}}
    sub_decision_tree = decision_tree[root_key]


    # Example: 0
    feature_index = header_names.index(root_key)

    for key in sub_decision_tree.keys():
        if test_dataset[feature_index] == key:
            if type(sub_decision_tree[key]).__name__ == 'dict':
                predict_label = predict(sub_decision_tree[key], header_names, test_dataset)
            else:
                predict_label = sub_decision_tree[key]

    return predict_label


def main():

  # Train
  dataset = [[0, 0, 0, 0, 'N'],
             [0, 0, 0, 1, 'N'],
             [1, 0, 0, 0, 'Y'],
             [2, 1, 0, 0, 'Y'],
             [2, 2, 1, 0, 'Y'],
             [2, 2, 1, 1, 'N'],
             [1, 2, 1, 1, 'Y']]
  header_names = ['outlook', 'temperature', 'humidity', 'windy']

  decision_tree = create_decision_tree(dataset, header_names)
  print("Train and get decision tree: {}".format(decision_tree))

  # Test
  header_names = ['outlook', 'temperature', 'humidity', 'windy']
  test_dataset = [2, 1, 0, 0]

  result = predict(decision_tree, header_names, test_dataset)
  print("Predict decision tree and get result: {}".format(result))

if __name__ == "__main__":
  main()


