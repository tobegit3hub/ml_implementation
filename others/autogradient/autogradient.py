#!/usr/bin/env python

import math
import numpy as np


class Op(object):
  def __init__(self):
    pass

  def forward(self):
    pass

  def grad(self):
    pass


class ConstantOp(Op):
  def __init__(self, x):
    self.x = x

  def forward(self):
    return self.x

  def grad(self):
    return 0


class CoefficientOp(Op):
  def __init__(self, x):
    self.x = x

  def forward(self):
    return self.x

  def grad(self):
    return self.x


class VariableOp(Op):
  def __init__(self, x):
    self.x = x

  def forward(self):
    return self.x

  def grad(self):
    return 1


def test_VariableOp():
  x = 10
  variable = VariableOp(x)
  print("X: {}, forward: {}, grad: {}".format(
      x, variable.forward(), variable.grad()))


class SquareOp(Op):
  def __init__(self, x):
    self.x = x

  def forward(self):
    return pow(self.x, 2)

  def grad(self):
    return 2 * self.x


def test_SquareOp():
  x = 10
  variable = SquareOp(x)
  print("X: {}, forward: {}, grad: {}".format(
      x, variable.forward(), variable.grad()))


class CubicOp(Op):
  def __init__(self, x):
    self.x = x

  def forward(self):
    return math.pow(self.x, 3)

  def grad(self):
    return 3 * math.pow(self.x, 2)


def test_CubicOp():
  x = 10
  variable = CubicOp(x)
  print("X: {}, forward: {}, grad: {}".format(
      x, variable.forward(), variable.grad()))


class AddOp(Op):
  def __init__(self, *ops):
    self.ops = ops

  def forward(self):
    result = 0
    for op in self.ops:
      result += op.forward()
    return result

  def grad(self):
    result = 0
    for op in self.ops:
      result += op.grad()
    return result


class MultipleOp(Op):
  def __init__(self, *ops):
    self.ops = ops

  def forward(self):
    result = 1
    for op in self.ops:
      result *= op.forward()
    return result

  def grad(self):
    result = 1
    for op in self.ops:
      result *= op.grad()
    return result


def main():
  x = 1

  # y = 3 * x**3 + 2 * x**2 + x + 10
  # y' =  9 * x**2 + 4 * x + 1

  first_item = MultipleOp(CoefficientOp(3), CubicOp(x))
  second_item = MultipleOp(CoefficientOp(2), SquareOp(x))
  third_item = VariableOp(x)
  forth_item = ConstantOp(10)
  y = AddOp(AddOp(first_item, second_item), third_item, forth_item)

  # Should be "X: 1, forward: 16.0, grad: 14.0"
  print("X: {}, forward: {}, grad: {}".format(x, y.forward(), y.grad()))


if __name__ == "__main__":
  main()
