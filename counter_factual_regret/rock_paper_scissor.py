#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random


class RockPaperScissorGame(object):
  """
  剪刀石头布游戏，通过regret matching算法收集历史比赛的regret，如果对手较大概率出一种，那么可以学习出应对这种的方法。
  
  原理是我们游戏后可以计算每个action的regret，每次都选regret最大的，计算regret的函数应该保证使用regret最大的以后正的regret越来越少，理想情况下没有正regret或者稳定就是达到纳什均衡点。
  
  Regret matching算法分为四步：
  第一步，随机选择action进行游戏。
  第二步，计算每次比赛每个action的regret，累计起来。
  第三步，选择累计regret最大的action，大概率情况下使用。
  第四步，重复进行游戏，累计每次比赛每个action的regret。
  第五步，最后如果要预测的话就使用正的累计regret来计算每个action的概率。
  """

  def __init__(self):
    # Initialize the constant values for the game
    self.ACTION_ROCK = 0
    self.ACTION_PAPER = 1
    self.ACTION_SCISSOR = 2

    self.ACTION_NUMBER = 3

    # Store the value of accumulative regret value for each action
    self.accumulative_regret_array = [0.0, 0.0, 0.0]
    self.accumulative_action_probability_array = [0.0, 0.0, 0.0]

  def get_action_probability(self):
    """
    返回当前用户应该执行的各个action的概率。
    
    原理是每次比赛后，记录每个action的regret，并且记录累加值，需要获取action的概率时就把累计regret为正的取出来，以这些正概率的占比作为选择action的概率，根据历史保证exploitation选历史regret大的这样历史regret增长变慢。
    注意，这里也累计了每次每个action的probability，用于以后预测时抛弃所有的regret信息，单纯考虑以前历史上每个action probability的均值。
    """

    action_probability = [0.0, 0.0, 0.0]
    total_positive_accumulative_regret = 0.0

    # Accumulate the positive historical regret
    for i in range(self.ACTION_NUMBER):
      if self.accumulative_regret_array[i] > 0:
        action_probability[i] = self.accumulative_regret_array[i]
      else:
        action_probability[i] = 0

      total_positive_accumulative_regret += action_probability[i]

    # Compute the probability for each action
    for i in range(self.ACTION_NUMBER):
      if total_positive_accumulative_regret > 0:
        # Positive means something to improve, follow the probability
        action_probability[i] = action_probability[
            i] / total_positive_accumulative_regret
      else:
        # For the first time or for the negative accumulative regret which means we can randomly choose
        action_probability[i] = 1.0 / self.ACTION_NUMBER

    for i in range(self.ACTION_NUMBER):
      self.accumulative_action_probability_array[i] += action_probability[i]

    return action_probability

  def get_action(self, action_probability):
    """
    给定每个action的probability，按照概率随机返回对应的action。    
    
    原理是先random一个0到1的数，落在任意一个probability区间内就返回对应的值，因为随机保证了exploration。
    """

    random_value = random.random()

    accumulative_probability = 0.0
    for i in range(3):
      accumulative_probability += action_probability[i]

      if random_value < accumulative_probability:
        # print("Choose from the action probability: {}, the action: {}".format(action_probability, i))
        return i

  def train(self):
    """
    开始训练regret matching算法来玩游戏。
    """

    iteration_number = 10000
    player2_action_probability = [0.4, 0.3, 0.3]

    for i in range(iteration_number):

      # 1. Get action probability from historical regret
      player1_action_probability = self.get_action_probability()

      # 2. Get action from action probability
      player1_action = self.get_action(player1_action_probability)
      player2_action = self.get_action(player2_action_probability)
      # print("Player1 action: {}, player2 action: {}".format(player1_action, player2_action))

      # 3. Compute regret for each action
      if player1_action == self.ACTION_ROCK and player2_action == self.ACTION_ROCK:
        action_rock_regret = 0
        action_paper_regret = 1
        action_scissor_regret = -1
      elif player1_action == self.ACTION_ROCK and player2_action == self.ACTION_PAPER:
        action_rock_regret = 0
        action_paper_regret = 1
        action_scissor_regret = 2
      elif player1_action == self.ACTION_ROCK and player2_action == self.ACTION_SCISSOR:
        action_rock_regret = 0
        action_paper_regret = -2
        action_scissor_regret = -1
      elif player1_action == self.ACTION_PAPER and player2_action == self.ACTION_ROCK:
        action_rock_regret = -1
        action_paper_regret = 0
        action_scissor_regret = -2
      elif player1_action == self.ACTION_PAPER and player2_action == self.ACTION_PAPER:
        action_rock_regret = -1
        action_paper_regret = 0
        action_scissor_regret = 1
      elif player1_action == self.ACTION_PAPER and player2_action == self.ACTION_SCISSOR:
        action_rock_regret = 2
        action_paper_regret = 0
        action_scissor_regret = 1
      elif player1_action == self.ACTION_SCISSOR and player2_action == self.ACTION_ROCK:
        action_rock_regret = 1
        action_paper_regret = 2
        action_scissor_regret = 0
      elif player1_action == self.ACTION_SCISSOR and player2_action == self.ACTION_PAPER:
        action_rock_regret = -2
        action_paper_regret = -1
        action_scissor_regret = 0
      elif player1_action == self.ACTION_SCISSOR and player2_action == self.ACTION_SCISSOR:
        action_rock_regret = 1
        action_paper_regret = -1
        action_scissor_regret = 0

      # 4. Accumulate the regret for each action
      self.accumulative_regret_array[0] += action_rock_regret
      self.accumulative_regret_array[1] += action_paper_regret
      self.accumulative_regret_array[2] += action_scissor_regret
      print(self.accumulative_regret_array)

  def get_final_action_probability(self):
    """
    返回最终每个action应该被执行的probability。

    原理是前面训练时已经计算了每次推荐的action probability，然后使用这些概率的累计值作为最终的预测值。
    """

    final_action_probability = [0.0, 0.0, 0.0]
    total_positive_accumulative_regret = 0.0

    for i in range(self.ACTION_NUMBER):
      total_positive_accumulative_regret += self.accumulative_action_probability_array[
          i]

    for i in range(self.ACTION_NUMBER):
      final_action_probability[i] = self.accumulative_action_probability_array[
          i] / total_positive_accumulative_regret

    return final_action_probability


def main():
  print("Start the rock paper scissor game")

  game = RockPaperScissorGame()
  game.train()
  final_action_probability = game.get_final_action_probability()

  print("The final probabilities of each action: {}".format(
      final_action_probability))


if __name__ == "__main__":
  main()
