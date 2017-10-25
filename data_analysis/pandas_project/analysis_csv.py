#!/usr/bin/env python

import numpy as np
import scipy as sp
import pandas as pd
import pprint


def main():
  # Load CSV file
  csv_file_path = "../data/train.csv"
  #csv_file_path = "../data/train_fe1.csv"
  dataset = pd.read_csv(csv_file_path)

  view_sample_dataset(dataset)
  print_dataset_info(dataset)
  print_features_info(dataset)


def view_sample_dataset(dataset):
  print("\n[Debug] Print the sample of the dataset: ")
  dataset_sample = dataset.head(1)
  print(dataset_sample)


def print_dataset_info(dataset):
  print("\n[Debug] Print the total number of the examples: ")
  example_number = len(dataset)
  print(example_number)

  print("\n[Debug] Print the info of the dataset: ")
  dataset_info = dataset.info()
  print(dataset_info)


def print_features_info(dataset):
  features_and_types = dataset.dtypes

  print("\n[Debug] Print the feature number: ")
  numberic_feature_number = 0
  not_numberic_feature_number = 0
  for feature_type in features_and_types:
    if feature_type == np.int16 or feature_type == np.int32 or feature_type == np.int64 or feature_type == np.float16 or feature_type == np.float32 or feature_type == np.float64 or feature_type == np.float128 or feature_type == np.double:
      numberic_feature_number += 1
    else:
      not_numberic_feature_number += 1
  print("Total feature number: {}".format(len(features_and_types)))
  print("Numberic feature number: {}".format(numberic_feature_number))
  print("Not numberic feature number: {}".format(not_numberic_feature_number))

  print("\n[Debug] Print the feature list of the dataset: ")
  print(features_and_types)

  print("\n[Debug] Print the feature presence: ")
  example_number = len(dataset)
  features_array = list(dataset.columns.values)
  for feature_name in features_array:
    feature_presence_number = len(dataset[feature_name][dataset[feature_name].notnull()])
    feature_presence_percentage = 100.0 * feature_presence_number / example_number
    # Example: "Age: 80.1346801347% (714 / 891)"
    print("{}: {}% ({} / {})".format(feature_name, feature_presence_percentage, feature_presence_number, example_number))

  print("\n[Debug] For numberic features, print the feature statistics: ")
  feature_statistics = dataset.describe()
  print(feature_statistics)

  top_k_number = 5
  print("\n[Debug] For all features, print the top {} values: ".format(top_k_number))
  for i in range(len(features_array)):
    feature_name = features_array[i]
    top_k_feature_info = dataset[feature_name].value_counts()[:top_k_number]
    print("\nFeature {} and the top {} values:".format(feature_name, top_k_number))
    print(top_k_feature_info)


if __name__ == "__main__":
  main()
