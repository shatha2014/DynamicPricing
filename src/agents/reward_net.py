# Adapted from:
# https://github.com/venkatacrc/Budget_Constrained_Bidding/
# which was originally from:
# https://github.com/udacity/deep-reinforcement-learning/blob/master/solution/dqn_agent.py

import numpy as np
import random
from collections import namedtuple, deque, defaultdict

from DynamicPricing.src.agents.model import Network

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5) #replay buffer size
BATCH_SIZE = 32 # minibatch size
LR = 1e-3 #learning rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RewardNet():
    """
    Interacts with and learns from the environment
    """

    def __init__(self, state_action_size, reward_size, seed):
        """
        Initialise a RewardNet object
        """
        print("reward 1")
        self.state_action_size = state_action_size
        print("reward 2")
        self.reward_size = reward_size
        print("reward 3")
        self.seed = random.seed(seed)

        #Reward Network
        self.reward_net =  Network(state_action_size, reward_size, seed).to(device)
        print("reward 4")
        self.optimizer = optim.Adam(self.reward_net.parameters(),lr=LR)
        print("reward 5")

        #Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        print("reward 6")
        # initialise the time step for update every UPDATE_EVERY steps
        self.t_step  = 0
        # Reward dict
        self.M  = defaultdict()

    def add(self, state_action, reward):
        # Save experience in replay memory
        self.memory.add(state_action, reward)

    def add_to_M(self, sa, reward):
        self.M[sa] = reward

    def get_from_M(self, sa):
        return (self.M.get(sa, 0))

    def step(self):
        # if enough samples are available in memory, get random subset and learn
        if len(self.memory) > 1:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state_action):
        sa = torch.from_numpy(state_action).double().unsqueeze(0).to(device)
        # self.reward_net.eval()
        # with torch.no_grad():
        #   action_values = self.reward_net(state)
        #self.reward_net.train()
        return self.reward_net(sa)

    def learn(self, experiences):
        """ Update value parameters using given batch of experience tuples
        :param experiences: (Tuple[torch.tensor]): tuple of (s, a, r, s', done) tuples
        :return:
        """
        state_actions, rewards= experiences

        # get expected reward values
        R_pred = self.reward_net(state_actions)

        # compute loss
        loss = F.mse_loss(R_pred, rewards)
        print("RewardNet loss = {}".format(loss))
        # minimise the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class ReplayBuffer:
    """ Fixed size buffer to store experience tuples
    """

    def __init__(self, buffer_size, batch_size, seed):
         """ Initialise a replay buffer
         :param buffer_size: maximum buffer size (int)
         :param batch_size: size of each training batch (int)
         :param seed: random seed (int)
         :return:none
         """
         self.memory = deque(maxlen=buffer_size)
         self.batch_size = batch_size
         self.experience = namedtuple("Experience", field_names=["state_action", "reward"])
         self.seed = random.seed(seed)

    def add(self, state_action, reward):
        """ Adds a new experience to the memory
        """
        self.memory.append(self.experience(state_action, reward))


    def sample(self):
        """
        Randomly sample a batch of experiences from memory
        """
        k = min(self.batch_size, len(self.memory))
        experiences = random.sample(self.memory, k=k)

        state_actions = torch.from_numpy(np.vstack([e.state_action for e in experiences if e is not None])).float64().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float64().to(device)

        return (state_actions, rewards)

    def __len__(self):
        """
        :return: Returns the current size of internal memory
        """
        return len(self.memory)