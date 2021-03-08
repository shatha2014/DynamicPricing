import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import pandas as pd
import configparser
import random
class PricingAuctions(gym.Env):
  """
  Add Description
  """
  def _load_config(self):
    """
    Parse the config file to read the data path
    """
    env_dir = os.path.dirname(__file__)
    #self.data_src= env_dir + '/sampledata.csv'
    self.data_src='/content/QSD Customer Segmentation v03.xlsx'

  def __init__(self):
    """
    Populates the orders and bids
    """
    self._load_config()
    self._step = 1
    #TODO: Refactor, read them from config as a list then a loop to populate them in the observation
    # use datetime for the date fields, there is a column of Agreed Liability (in w) that i ignored currently
    df = pd.read_excel('/content/QSD Customer Segmentation v03.xlsx' ,header=0, converters= { 'PL': str, 'Customer': str, 'Order Number': str, 'Order Entry Date':str, 'Customer Wish Date':str, 'Confirmed Delivery Date':str, 'Requested Order Lead Time (in d)':float, 'Confirmed Order Lead Time (in d)':float, 'Order quantity': float, 'Product Information': str, 'Sales product': lambda s: float(s.replace('S', '')), 'Importance of order in €': float})
    df.rename(columns={'Customer': 'customer_id', 'Order Number': 'order_id', 'Order Entry Date':'orderentry_date',
                                      'Customer Wish Date': 'customer_wish_date', 'Confirmed Delivery Date':'confirmed_delivery_date',
                                      'Requested Order Lead Time (in d)':'customer_requestedLT', 'Confirmed Order Lead Time (in d)': 'confirmed_orderLT',
                                      'Order quantity': 'order_quantity', 'Product Information ':'product_id', 'Sales product':'sales_product',
                                      'Importance of order in €': 'order_importance'}, inplace=True)
    #temporary
    df['customer_sensitivity'] = 0.9
    df['bid_price_rl'] = 0.0
    df['customer_accepted_price'] = 0.0
    df['auction_type'] = 'FIRST_PRICE'
    df[df['customer_id'] == 'C158']['customer_sensitivity'] = 0.2 #TODO
    self.bid_requests = df[df['customer_id'] == 'C158']
    self.total_bids = len(self.bid_requests)

  def _get_observation(self, bid_req):
    """
    add description
    """
    observation ={}
    #populate everything except bidding prices and accepted prices
    #TODO double check hte fields and refactor not sure about confirmed dates
    if bid_req is not None:
      observation['customer_id'] = bid_req['customer_id']
      observation['order_id'] = bid_req['order_id']
      observation['orderentry_date'] = bid_req['orderentry_date']
      observation['customer_wish_date'] = bid_req['customer_wish_date']
      observation['confirmed_delivery_date'] = bid_req['confirmed_delivery_date']
      observation['customer_requestedLT'] = bid_req['customer_requestedLT']
      observation['confirmed_orderLT'] = bid_req['confirmed_orderLT']
      observation['order_quantity'] = bid_req['order_quantity']
      observation['product_id'] = bid_req['product_id']
      observation['sales_product'] = bid_req['sales_product']
      observation['order_importance'] = bid_req['order_importance']
      observation['customer_sensitivity'] = bid_req['customer_sensitivity']
    return observation

  def _bid_state(self, bid_req):
    """
    add description
    """
    self.auction_type = bid_req['auction_type']
    self.bidprice_rl = bid_req['bid_price_rl']
    self.customer_acceptedprice = bid_req['customer_accepted_price']
    self.customer_sensitivity = bid_req['customer_sensitivity']
    self.sales_product = bid_req['sales_product']
    self.order_quantity = bid_req['order_quantity']
    self.orderentry_date = bid_req['orderentry_date']
    self.confirmed_delivery_date = bid_req['confirmed_delivery_date']

  def reset(self):
    """
    Reset the environment
    """
    self._step = 1
    bid_req = self.bid_requests.iloc[self._step]
    self._bid_state(bid_req)
    # observation, reward, cost, done
    return self._get_observation(bid_req), 0.0, 0.0, False

  def step(self, action):
    """
    action: bid response (bid price)
    Reward is calculated using the difference between bid price and the price that
    the customer accepted (from the RL agent or other pricing agents)
    """
    done = False
    r = 0.0 #immediate reward
    c = 0.0 # cost of the bid price

    #r = self.customer_sensitivity
    #r= random.uniform(0.2,0.9)
    r = ((self.sales_product * self.order_quantity) + (self.customer_sensitivity * action)) - action
    c = action

    next_bid = None
    if self._step < self.total_bids -1:
      next_bid = self.bid_requests.iloc[self._step]
      self._bid_state(next_bid)
    else:
      done = True

    self._step += 1

    return self._get_observation(next_bid), r, c, done


  def render(self, mode='human', close=False):
    pass

  def close(self):
    pass