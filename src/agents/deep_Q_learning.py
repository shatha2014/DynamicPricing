# Adapted from:
# https://github.com/venkatacrc/Budget_Constrained_Bidding/
# which was originally from:
# https://github.com/udacity/deep-reinforcement-learning/blob/master/solution/dqn_agent.py

import numpy as np
import random
from collections import namedtuple, deque


from DynamicPricing.src.agents.model import Network

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5) #replay buffer size
BATCH_SIZE = 32 # minibatch size
GAMMA = 1.0 # discount factor
TAU = 1e-3 #for soft update of target parameters
LR = 1e-3 #learning rate
UPDATE_EVERY = 4 # how often to update the target network

# run on GPU or CPU
device = torch.device("cudo:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """
    DQN Agent that Interacts with the environment
    """
    def __init__(self, state_size, action_size, seed):
        """
        initialise an agent object
        :param state_size: dimension of state
        :param action_size:dimension of actions
        :param seed: random seed
        :return: none
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q Network
        self.qnetwork_local = Network(state_size, action_size,seed).to(device)
        self.qnetwork_target = Network(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE,BATCH_SIZE,seed)
        # initialise time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def step(self, state,action,reward, next_state,done):
        """
        Receive from the environment and save experience in replay memory
        :param state: current observation
        :param action: performed action
        :param reward: recevied reward
        :param next_state: next state
        :param done: epsiode done or not
        :return:none
        """

        # save experience in replay memory
        self.memory.add(state,action,reward,next_state, done)

        # learn every update_every time steps
        self.t_step = (self.t_step + 1 ) % UPDATE_EVERY
        if self.t_step == 0:
            # if enough samples are available in memory, get random subset and learn
            # if len(self.memory) > Batch_size
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)


    def act(self, state, eps=0.):
        """
        :param state: current state to be provided as input to the model
        :param eps: float - epsilon value (epsilon greedy action selection)
        :return: Returns actions for given state as per current policy
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples
        :param experiences: (Tuple[torch.tensor]): tuple of (s, a, r, s', done) tuples
        :param gamma: float - discount factor
        :return: none
        """

        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_target_next * (1- dones))

        # Get expected Q values from local model
        print("states : ")
        print(states)
        print(states.dtype)
        print("actions: ")
        print(actions)
        print(actions.dtype)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute the loss
        loss = F.mse_loss(Q_expected, Q_targets)
        print("DQN loss = {}".format(loss))
        # Minimise the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: Weights will be copied from local network to
        :param target_model: target network
        :param tau: TAU parameter
        :return: none
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy(tau*local_param.data + (1.0*tau)* target_param.data)

class ReplayBuffer():
    """ Fixed size buffer to store experience tuples
    """

    def __init__(self, buffer_size, batch_size, seed):
        """
        Initialise a ReplayBuffer object
        :param buffer_size: (int) maximum size of the buffer
        :param batch_size: (int) size of each training batch
        :param seed: random seed
        :return: none
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state","action","reward","next_state", "done"])
        self.seed =random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the memory
        """
        self.memory.append(self.experience(state, action,reward, next_state,done))

    def sample(self):
        """
        Randomly sample a batch of experiences from memory
        """
        k = min(self.batch_size, len(self.memory))
        experiences = random.sample(self.memory, k=k)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        :return: Returns the current size of the internal memory
        """
        return len(self.memory)





