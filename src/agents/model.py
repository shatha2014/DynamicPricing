"""
# Adapted from:
# https://github.com/venkatacrc/Budget_Constrained_Bidding/
originally from: https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/model.py
The original code was modified to add one more hidden layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    """ Actor policy model
    """

    def __init__(self, state_size, action_size, seed, fc1_units = 100, fc2_units=100, fc3_units=100):
        """
        Initialisation of parameters and building the model
        :param state_size: dimension of the observation state
        :param action_size: dimension of the actions ==  q-values representing the actions
        :param seed: random seed value
        :param fc1_units: number of hidden units in fully connected layer 1
        :param fc2_units: number of hidden units in fully connected layer 2
        :param fc3_units: number of hidden units in fully connected layer 3
        :return: none
        """
        super(Network,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)



    def forward(self, state):
        """
        :param state: builds the network that takes a state as input and produces actions approximated values
        :return: action predicted q values
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
