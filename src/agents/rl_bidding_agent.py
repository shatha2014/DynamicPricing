import sys, os
#sys.path.append(os.getcwd() +'/src/auction')
import gym
#from DynamicPricing.src.auction.envs.pricing_auction_env import PricingAuctions

import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import configparser
from DynamicPricing.src.agents.deep_Q_learning import Agent
from DynamicPricing.src.agents.reward_net import RewardNet


class RlBidAgent():

    def __load_config(self):
        #TODO change to read from config
        # timesteps #TODO change value
        self.T = 24
        self.STATE_SIZE = 2 #tTODO
        # State:
        # Customer ID -- personlisation
        # Total number of orders for this customer
        # Original time of order
        # Saved percentage
        # Order quantity
        # sales product
        # order importance
        # customer sensitivity
        # past order performance -- toDO
        self.STATE_SIZE = 8
        self.ACTION_SIZE = 2 #TODO

    def __init__(self):
        self.__load_config()
        # Control parameter used to scale bid price #TODO
        self.BETA = []
        self.eps_start = 0.0
        self.eps_end = 0.0
        self.anneal = 0.0
        self._reset_episode()

        # DQN Network to learn Q function
        #TODO
        self.dqn_agent = Agent(state_size = 8, action_size = 2, seed =0)
        # Reward Network to reward function
        #TODO
        self.reward_net = RewardNet(state_action_size = 8, reward_size = 1, seed =0 )
        self.dqn_state = None
        self.dqn_action = 0 #TODO
        self.dqn_reward = 0
        # Reward dictionary
        self.reward_dict = {}
        self.S = []
        self.V = 0
        self.total_wins = 0
        self.total_rewards = 0.0
        self.bid_count = 0 #TODO -- for episode done

        self.customer_id = 0
        self.order_id = 0
        self.orderentry_date = None
        self.customer_wish_date = None
        self.confirmed_delivery_date = None
        self.customer_requestedLT =0
        self.confirmed_orderLT_A = 0
        self.order_quantity = 0
        self.product_id = 0
        self.sales_product = 0.0
        self.order_importance = 0.0
        self.customer_sensitivity = 0.0


    def _reset_episode(self):
        """
        Function to reset the state when episode changes
        """
        self.t_step = 0 #1. t: the current time step
        #TODO
        #self.customer_requestedLT = 0
        #self.confirmed_orderLT = 0
        #self.order_quantity = 0
        #self.sales_product = 0
        #self.order_importance = 0
        #self.customer_sensitivity = 0
        #self.customer_acceptance = 0 #TODO --> how many times the customer accepted proposal from this pricing agent

        self._reset_step()
        #TODO -- day and time
        self.wins_e = 0
        self.eps = self.eps_start
        self.V = 0

    def _update_step(self):
        """
        Function to call update the state with every bid request
        """
        self.t_step += 1
        #TODO check the other fields
        self.customer_acceptance  = self.wins_t / self.bids_t

    def _reset_step(self):
        """
        Function to call every time a new time step is entered
        """
        self.rewards_t = 0.
        self.cost_t = 0.
        self.wins_t= 0.
        self.bids_t = 0.
        self.eps = max(self.eps_start - self.anneal * self.t_step, 0.05) #TODO check this

    def _update_reward_cost(self, reward, cost):
        """
        Function to update reward and action to compute the cumulative reward
        and cost within the given step
        """
        self.reward_t += reward
        self.cost_t += cost
        self.bids_t += 1
        self.total_rewards += reward

    def _get_state(self,state):
        """
        Returns the state that will be used for the DQN state
        """
        #self.t_step,self.proposed_price, #self.customer_acceptance,
        return np.asarray([ state['customer_id'],
                            state['orderentry_date'],
                            state['confirmed_delivery_date'],
                            state['customer_requestedLT'],
                            state['confirmed_orderLT_A'],
                            state['order_quantity'],
                            state['sales_product'],
                            state['order_importance'],
                            state['customer_sensitivity'],
                            self.reward_t
        ])

    def act(self, state, reward, cost):
        """
        This function gets called with every bid request.
        By looking at the order quantity, sales product, order importance
        customer sensitivity, requested lead time, confirmed lead time and
        customer acceptance, returns the bid request cost based on the scaled
        version of the bid price using DQN agent output
        """
        self.bid_count += 1
        episode_done = (self.bid_count % 10 == 0) #each 10 requests ? #TODO

        # Within the epsiode
        if not episode_done:
            self._update_step() #TODO  assuming that each request is a step
            # sample a mini-batch and perform grad descent step
            self.reward_net.step()
            dqn_next_state = self._get_state(state)
            a_beta = self.dqn_agent.act(dqn_next_state, eps=self.eps) #TODO
            sa = np.append(self.dqn_state, self.dqn_action)
            rnet_r = float(self.reward_net.act(sa)) #state -- produce reward
            # call agent step
            self.dqn_agent.step(self.dqn_state, self.dqn_action, rnet_r, dqn_next_state, episode_done)
            self.dqn_state = dqn_next_state
            self.dqn_action = a_beta
            print(dqn_next_state, a_beta)
            #TODO
            self._reset_step()
            self._update_reward_cost(reward, cost)
            self.V += self.rewards_t
            self.S.append(self.dqn_state, self.dqn_action)
        # episode changes
        else:
            for(s, a) in self.S:
                sa = tuple(np.append(s,a))
                max_r= max(self.reward_net.get_from_M(sa), self.V)
                self.reward_net.add_to_M(sa, max_r)
                self.reward_net.add(sa, max_r)
            print("Total Accepted Proposals won with Customer ={}".format(self.wins_e))
            self.total_wins += self.wins_e
            self._reset_episode()
            self._update_reward_cost(reward,cost)

        # action --> bid amount
        # Should be computed using a formula
        # Proposed price should be based on the original price, order size, order importance, customer sensitivity
        if cost > 0: #cost from previous step
            self.wins_e += 1
            self.wins_t += 1
        # action - propose adjusted price or propose original price
        action = self.dqn_action
        #TODO action space -- equal to the number of bids
        return action

    def done(self):
        return False #TODO decide how it is done






