#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random


class Node(object):
  """
  表示游戏种某一个状态，包含了用户的手牌、前面的action选择序列，还有这种状态下每个action的累计regret。
  注意，这里的card_actionsequence_string代表了论文中的information set。
  """

  def __init__(self):
    self.ACTION_NUMBER = 2

    self.accumulative_regret_array = [0.0, 0.0]
    self.accumulative_action_probability_array = [0.0, 0.0]

    self.card_actionsequence_string = ""

  def get_action_probability(self, read_probability=1.0):
    """
    返回各个action的选取probability，还有更新每个action累计的probability，如果输入read_probability则在累计时乘以作为权重。
    
    action选取概率的计算原理regret matching算法，根据每个action的累计regret，选择所有正值，归一化保证和为0后返回每个action的probability。
    注意，在累加所有action的probability时，如果传入read_probability就作为权重相乘，这是CFR算法考虑到每次到达这个状态计算action probability时也有一定概率。
    """

    action_probability = [0.0, 0.0]
    total_positive_accumulative_regret = 0.0

    # Compute the sum of all positive regret
    for i in range(self.ACTION_NUMBER):
      if self.accumulative_regret_array[i] > 0:
        action_probability[i] = self.accumulative_regret_array[i]
      else:
        action_probability[i] = 0.0

      total_positive_accumulative_regret += action_probability[i]

    # Divide by the sum of all positive regret
    for i in range(self.ACTION_NUMBER):
      if total_positive_accumulative_regret > 0:
        action_probability[i] = action_probability[
            i] / total_positive_accumulative_regret
      else:
        action_probability[i] = 1.0 / self.ACTION_NUMBER

    # Accumulate the action probability with read probability as weight, which is used for getting final action probability
    for i in range(self.ACTION_NUMBER):
      self.accumulative_action_probability_array[i] += action_probability[
          i] * read_probability

    return action_probability

  def get_final_action_probability(self):
    """
    返回各个action的选取probability。
    
    原理是根据历史累计的每个action的probability，归一化和为1后直接返回。
    """

    final_action_probability = [0.0, 0.0]
    total_positive_accumulative_regret = 0.0

    # Compute the sum of all action probabilities
    for i in range(self.ACTION_NUMBER):
      total_positive_accumulative_regret += self.accumulative_action_probability_array[
          i]

    # Divide by the sum of all action probabilities
    for i in range(self.ACTION_NUMBER):
      if total_positive_accumulative_regret > 0.0:
        final_action_probability[
            i] = self.accumulative_action_probability_array[
                i] / total_positive_accumulative_regret
      else:
        final_action_probability[i] = 1.0 / self.ACTION_NUMBER

    return final_action_probability

  def get_card_actionsequence_string(self):
    return self.card_actionsequence_string

  def set_card_actionsequence_string(self, card_actionsequence_string):
    self.card_actionsequence_string = card_actionsequence_string

  def __str__(self):
    final_action_probability = self.get_final_action_probability()
    return "The card and action sequence: {}, the final action probability: {}".format(
        self.card_actionsequence_string, final_action_probability)


class KuhnPokerGame(object):
  """
  表示Kuhn扑克游戏，使用CFR算法进行训练和优化，详情参考 https://en.wikipedia.org/wiki/Kuhn_poker 。
  
  游戏需要包含所有游戏状态，也就是Node的集合，Node也包含了总regret和所有action probability。
  需要支持train()函数用户训练模型，还有实现cfr()函数用于递归计算每个Node状态下预期的payoff，并且更新这个Node下每个action对应的累计regret。
  """

  def __init__(self):
    # Initialize the constant values of the game
    self.ACTION_PASS = 0
    self.ACTION_BET = 1
    self.ACTION_NUMBER = 2

    # No need to store the terminal Nodes
    self.node_map = {}

  def shuffle_list(self, original_list):
    shuffled_list = [item for item in original_list]
    random.shuffle(shuffled_list)
    return shuffled_list

  def train(self):
    iteration_number = 10000
    validate_iteration_number = 100
    cards = [1, 2, 3]

    # Start training
    for i in range(iteration_number):
      cards = self.shuffle_list(cards)
      self.cfr(cards, "", 1, 1)

    # Start validating
    total_validated_payoff = 0.0
    for i in range(validate_iteration_number):
      cards = self.shuffle_list(cards)
      this_episode_payoff = self.cfr(cards, "", 1, 1)

      total_validated_payoff += this_episode_payoff
      #print("This episode cards: {}, payoff: {}".format(cards, this_episode_payoff))

    average_validate_payoff = total_validated_payoff / validate_iteration_number
    print("The average validate payoff: {}, iteraction: {}".format(
        average_validate_payoff, iteration_number))

  def cfr(self, cards, action_sequence, player0_reach_probability,
          player1_reach_probability):
    """
    Counter factual regret minimization算法，输入随机选择的cards数组，之前已经执行过的action序列，还有分别是player0和player1的到达这个状态的reach probability，返回这个Node状态下的期望payoff。

    例如，传入action_sequence为"bp"则直接返回payoff为+1，如果传入action_sequence为空字符串则从头开始游戏并返回某种选择下的payoff，同时会更新累计的regret有助于下一次选择。

    CFR（复盘后悔值算法）原理是，先给定一个游戏状态（Node中包含两个玩家的手牌和之前的action sequence信息），计算这个状态下的payoff值。
    如果action sequence表示游戏到达了terminal状态，那么根据游戏规则返回对应确定的payoff值即可。
    如果处于非terminal状态，那么要加载这个游戏状态（也就是Node）每个action（也就是pass和bet）的累计regret值，累计正的regret值进行归一化（也就是regret matching）得到action probability。
    因为还没到terminal状态，所以可以遍历所有action执行下面的计算，确定某一个action后，递归调用CFR用户计算下一个状态（也就是对手的状态）下的期望payoff，乘以负一表示自己在这个action下的期望payoff，然后累加所有action的总期望作为这个Node的payoff。
    注意，因为每一个action都有action probability，每次递归时都需要对当前自己这个用户的reach probability乘以这个action的action probability进行更新。
    然后我们知道在这个Node下每个action的payoff和总payoff，那么每个action的regret就是总payoff减去每个action的payoff，这个regret会累加记录到这个Node的信息中方便以后使用regret matching选择regret最大的action。
    注意，这种regret是在对手采取上一步某种action的时候我们累加自己的所有action求出来的，所以我们应该在regret上乘以对手的reach probability。
    这样我们在游戏的过程，不断尝试选择所有action来复盘执行，把返回的payoff加上这种状态的reach probability计算并更新这种状态下的所有action的累计regret，以后就可以根据累计regret计算出action probability。
    
    Parames:
      cards: 可能为[1, 2, 3]或者是[2, 3, 1]
      action_sequence: 可能为中间状态（"", "b", "p", "pb"），或者最终状态（"pp", "pbp", "pbb", "bp", "bb"） 
      player0_reach_probability: 一开始是1.0，会乘以当前用户选择的action probability而变化。
      player1_reach_probability: 同上。
    """

    # 1. Analysis the played actions, get the current user and it's card
    play_times = len(action_sequence)
    current_player = play_times % 2
    opponent_player = 1 - current_player
    current_player_card = cards[current_player]
    opponent_player_card = cards[opponent_player]

    # 2. If has played 2 actions, check if it should terminate or not
    if play_times >= 2:

      # If it's player0 pass and player1 pass, return 1 for the better one
      if action_sequence == "pp":
        if current_player_card > opponent_player_card:
          return 1
        else:
          return -1

      # If it's player0 bet and player1 bet, return 1 for the better one
      if action_sequence == "bb" or action_sequence == "pbb":
        if current_player_card > opponent_player_card:
          return 2
        else:
          return -2

      # If someone passes, the player with betting should win and return 1
      if action_sequence == "bp" or action_sequence == "pbp":
        return 1
      """
      # TODO: "bp" should return 1 but "pbp" should return -1
      if action_sequence == "bp":
        return 1
        
      if action_sequence == "pbp":
        return -1
      """

    # Record the Node structure
    card_actionsequence_string = "{}{}".format(current_player_card,
                                               action_sequence)

    # 3. Add the new node in the dictionary structure
    if card_actionsequence_string in self.node_map:
      current_node = self.node_map[card_actionsequence_string]
    else:
      node = Node()
      node.set_card_actionsequence_string(card_actionsequence_string)
      self.node_map[card_actionsequence_string] = node
      current_node = node

    # 4. Get the action probability
    if current_player == 0:
      reach_probability = player0_reach_probability
    elif current_player == 1:
      reach_probability = player1_reach_probability

    action_probability = current_node.get_action_probability(reach_probability)

    # 5. Compute the payoff for each action
    node_action_payoff_array = [0.0, 0.0]
    node_total_payoff = 0.0

    for i in range(self.ACTION_NUMBER):
      if i == 0:
        action = "p"
      elif i == 1:
        action = "b"

      # Add the new choosen action for the action sequence
      next_action_sequence = action_sequence + action

      if current_player == 0:
        next_player0_weight = player0_reach_probability * action_probability[i]
        next_player1_weight = player1_reach_probability
      elif current_player == 1:
        next_player0_weight = player0_reach_probability
        next_player1_weight = player1_reach_probability * action_probability[i]

      node_action_payoff_array[i] = -1.0 * self.cfr(
          cards, next_action_sequence, next_player0_weight,
          next_player1_weight)
      node_total_payoff += action_probability[i] * node_action_payoff_array[i]

    for i in range(self.ACTION_NUMBER):
      regret = node_action_payoff_array[i] - node_total_payoff

      if current_player == 0:
        modified_regret = regret * player1_reach_probability
      elif current_player == 1:
        modified_regret = regret * player0_reach_probability

      current_node.accumulative_regret_array[i] += modified_regret

    return node_total_payoff

  def print_nodes(self):
    for node in self.node_map.values():
      print(node)


def main():
  print("Start the kuhn poker game")
  game = KuhnPokerGame()
  game.train()
  game.print_nodes()


if __name__ == "__main__":
  main()
