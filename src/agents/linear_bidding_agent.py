import sys

"""
"""
class LinearBiddingAgent():
  def _load_config(self):
    """
    """
    self.budget = 10
    self.target_value = 1

  def __init__(self):
    self._load_config()
    self.wins_e = 0 #wins in each episode
    self.total_wins =0 #total wins
    self.total_rewards = 0.0
    self.budget_spend = 0

  def act(self,state,reward, cost):
    # decide how the episode will be done
    # simply strategy -- 10% increase on price
    val = state['sales_product'] * state['order_quantity']
    action = val + val * 0.10

    if reward > 0.5:
      self.wins_e += 1
      self.total_wins += 1
      self.total_rewards += 1

    self.budget_spend += (cost / 1e9)

    #action = min(state['customer_sensitivity'] * action, action)
    return action

  def done(self):
    return self.budget <= self.budget_spend

