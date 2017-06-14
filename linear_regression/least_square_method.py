#!/usr/bin/env python


def main():
  print("Start training")

  x_array = [1.0, 2.0, 3.0, 4.0, 5.0]
  # y = 2x + 3
  y_array = [5.0, 7.0, 9.0, 11.0, 13.0]

  instance_number = len(x_array)

  learning_rate = 0.01
  epoch_number = 100

  w = 1.0
  b = 1.0

  for epoch_index in range(epoch_number):

    w_grad = 0.0
    b_grad = 0.0
    loss = 0.0

    for i in range(instance_number):
      x = x_array[i]
      y = y_array[i]

      w_grad += -2.0 / instance_number * x * (y - w * x - b)
      b_grad += -2.0 / instance_number * (y - w * x - b)

      loss += 1.0 / instance_number * pow(y - w * x - b, 2)

    w -= learning_rate * w_grad
    b -= learning_rate * b_grad
    print("Epoch is: {} w is: {}, w is: {}, loss is: {}".format(
        epoch_index, w, b, loss))


if __name__ == "__main__":
  main()
