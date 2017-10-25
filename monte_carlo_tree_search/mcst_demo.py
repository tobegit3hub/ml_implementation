#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import random
import numpy as np

AVAILABLE_CHOICES = [1, -1, 2, -2]
AVAILABLE_CHOICE_NUMBER = len(AVAILABLE_CHOICES)
MAX_TURN_INDEX = 10


class State(object):
  """
  记录每个搜索树Node的状态信息，包含当前节点，。
  
  支持判断当前状态是否达到游戏结束状态。
  
  支持从Action集合中取出操作，然后通过S和A得到下一个状态S‘。
  """

  def __init__(self):
    self.current_value = 0.0
    self.current_turn_index = 0
    self.cumulative_choices = []

  def get_current_value(self):
    return self.current_value

  def set_current_value(self, value):
    self.current_value = value

  def get_current_turn_index(self):
    return self.current_turn_index

  def set_current_turn_index(self, turn):
    self.current_turn_index = turn

  def get_cumulative_choices(self):
    return self.cumulative_choices

  def set_cumulative_choices(self, choices):
    self.cumulative_choices = choices

  def is_terminal(self):
    if self.current_turn_index == MAX_TURN_INDEX:
      return True
    else:
      return False

  def reward(self):
    return -abs(1 - self.current_value)

  def get_next_state_with_random_choice(self):
    random_choice = random.choice([choice for choice in AVAILABLE_CHOICES])

    next_state = State()
    next_state.set_current_value(self.current_value + random_choice)
    next_state.set_current_turn_index(self.current_turn_index + 1)
    next_state.set_cumulative_choices(self.cumulative_choices +
                                      [random_choice])

    return next_state

  def __eq__(self, other):
    if hash(self) == hash(other):
      return True
    return False


class Node(object):
  def __init__(self):
    """
    蒙特卡洛树搜索的Node结构，包含了父节点和直接点等信息，还有用于计算UCB的遍历次数和reward值。
    
    """

    self.parent = None
    self.children = []

    self.visit_times = 0
    self.reward_value = 0.0

    self.state = None

  def set_state(self, state):
    self.state = state

  def get_state(self):
    return self.state

  def set_parent(self, parent):
    self.parent = parent

  def get_children(self):
    return self.children

  def get_visit_times(self):
    return self.visit_times

  def set_visit_times(self, times):
    self.visit_times = times

  def visit_times_add_one(self):
    self.visit_times += 1

  def get_reward_value(self):
    return self.reward_value

  def set_reward_value(self, value):
    self.reward_value = value

  def reward_value_add_n(self, n):
    self.reward_value += n

  def is_all_expand(self):
    if len(self.children) == AVAILABLE_CHOICE_NUMBER:
      return True
    else:
      return False

  def add_child(self, sub_node):
    sub_node.set_parent(self)
    self.children.append(sub_node)


def tree_policy(node):
  """
  蒙特卡洛树搜索的搜索阶段，从当前节点找到下一个值得探索的节点，基本策略是先找未被探索的，然后找UCB最大的。
  """

  while node.get_state().is_terminal() == False:
    if node.is_all_expand():
      node = best_child(node, True)
    else:
      node = expand(node)
  return node


def default_policy(state):
  """
  已经找到了最下面的子节点，随机执行Action得到新的状态和子节点。
  """

  new_state = state
  while new_state.is_terminal() == False:
    new_state = state.get_next_state_with_random_choice()

  return new_state.reward()


def expand(node):
  """
  在原来节点上拓展一个新的节点并且返回。
  """

  tried_sub_node_states = [node.get_state() for node in node.get_children()]

  #node.get_state()

  new_state = node.get_state().get_next_state_with_random_choice()

  while new_state in tried_sub_node_states:
    new_state = node.get_state().get_next_state_with_random_choice()

  sub_node = Node()
  sub_node.set_state(new_state)

  node.add_child(sub_node)

  return sub_node


def best_child(node, is_exploration):
  """
  使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，如果是预测阶段直接选择当前得分最高的。
  """

  # TODO: Use the min float value
  best_score = -sys.maxsize
  best_sub_node = None

  for sub_node in node.get_children():

    if is_exploration:
      C = 1 / math.sqrt(2.0)
    else:
      C = 0.0

    # UCB = reward / times + C * sqrt(ln(2 * total_times) / times)
    left = sub_node.get_reward_value() / sub_node.get_visit_times()
    right = C * math.sqrt(
        math.log(2.0 * node.get_visit_times() / sub_node.get_visit_times()))
    score = left + right

    if score > best_score:
      best_sub_node = sub_node

  return best_sub_node


def backup(node, reward):
  """
  将reward反馈到此节点的所有父节点上。  
  """

  while node != None:
    node.visit_times_add_one()
    node.reward_value_add_n(reward)
    node = node.parent


def monte_carlo_tree_search(root_node):
  """
  蒙特卡洛树搜索算法，传入一个根节点，返回下一个  

  """

  computation_budget = 100

  for i in range(computation_budget):

    last_node = tree_policy(root_node)
    reward = default_policy(last_node.get_state())
    backup(last_node, reward)

  best_node = best_child(root_node, False)

  #import ipdb;ipdb.set_trace()
  return best_node


def main():
  print("Start mcst demo")

  init_state = State()
  init_node = Node()
  init_node.set_state(init_state)

  current_node = init_node
  for i in range(10):
    current_node = monte_carlo_tree_search(current_node)
    print("Tree level: {}".format(i))


if __name__ == "__main__":
  main()
