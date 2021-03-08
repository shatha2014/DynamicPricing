import sys
from datetime import datetime


"""
Linear Pricing Algorithm - This pricing strategy is solely based on the
relation between faster delivery to standard delivery time - So a 10% faster
delivery time corresponds to a 10% price premium
Formula is: Price(premium) = min{max[(1+ (Standard delivery time  - confirmed lead time)/standard delivery time) * original price; original price]; maximum price}
Maximum price is double the original price
Might be too high for sensitive customers
"""
class LinearBiddingAgent():
  def _load_config(self):
    """
    Add configurations if needed
    """

  def __init__(self):
    self._load_config()
    self.wins_e = 0 #wins in each episode
    self.total_wins =0 #total wins
    self.total_rewards = 0.0
    self.bid_count = 0.0


  def act(self,state,reward, cost):
    # Formula written at the top of the file

    # Step 1: Calculate number of days between order entry date and
    # confirmed delivery date
    confirmed_lead_time = state['confirmed_orderLT']
    order_length_days = self.days_between(state['orderentry_date'], state['confirmed_delivery_date']) + confirmed_lead_time
    print("order entry date is {}".format(state['orderentry_date']))
    print("confirmed delivery date is {}".format(state['confirmed_delivery_date']))
    print("order length in days is {}".format(order_length_days))
    print("confirmed lead time is {}".format(confirmed_lead_time))
    original_price = state['order_quantity'] * state['sales_product']
    print("original price is {}".format(original_price))
    maximum_price = original_price * 2
    print("maximum price is {}".format(maximum_price))
    percentage = confirmed_lead_time/order_length_days
    print("percentage is {}".format(percentage))
    calculated_price = max( ((1+ percentage) * original_price) , original_price)
    print("calculated price is {}".format(calculated_price))
    price_premimum = min(calculated_price, maximum_price)
    print("price premium is {}".format(price_premimum))

    action = price_premimum

    if reward > 0:
      self.wins_e += 1
      self.total_wins += 1
      self.total_rewards += 1

    self.bid_count += 1

    return action

  def done(self):
    return self.bid_count < 7000

  def days_between(self,d1, d2):
    d1 = datetime.strptime(d1, "%m.%Y")
    d2 = datetime.strptime(d2, "%m.%Y")
    return abs((d2 - d1).days)

