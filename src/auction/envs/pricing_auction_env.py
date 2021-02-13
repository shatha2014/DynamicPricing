import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import pandas as pd
import configparser
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
    self.data_src='/content/DynamicPricing/data/bidding_prices_test.csv'

  def __init__(self):
    """
    Populates the orders and bids
    """
    self._load_config()
    self._step = 1
    #TODO: Refactor, read them from config as a list then a loop to populate them in the observation
    fields =[
             'customer_id',
             'order_id',
             'orderentry_date',
             'customer_requestedLT',
             'confirmed_orderLT_A',
             'order_quantity',
             'product_id',
             'sales_product',
             'order_importance',
             'bidprice_A',
             'customer_sensitivity',
             'customer_acceptedprice',
             'auction_type'
    ]
    self.bid_requests = pd.read_csv(self.data_src, sep=",")
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
      observation['customer_requestedLT'] = bid_req['customer_requestedLT']
      observation['confirmed_orderLT_A'] = bid_req['confirmed_orderLT_A']
      observation['order_quantity'] = bid_req['order_quantity']
      observation['product_id'] = bid_req['product_id']
      observation['sales_product'] = bid_req['sales_product']
      observation['auction_type' ] = bid_req['auction_type']
    return observation

  def _bid_state(self, bid_req):
    """
    add description
    """
    self.auction_type = bid_req['auction_type']
    self.bidprice_A = bid_req['bidprice_A']
    self.customer_acceptedprice = bid_req['customer_acceptedprice']
    self.customer_sensitivity = bid_req['customer_sensitivity']
    self.order_importance = bid_req['order_importance']

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

    r = self.customer_sensitivity
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