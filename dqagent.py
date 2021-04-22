import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from dueling_model import QNetwork
from experience import PriorityReplayBuffer

LR = 5e-4               # learning rate
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
UPDATE_EVERY = 4        # how often to update the network
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
OPTIMIZER_LR = 0.9999

class Agent:
    def __init__(self, state_size, action_size, seed):
        # Configs
        self.state_size = state_size
        self.action_size = action_size

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed=seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed=seed)
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=LR)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, OPTIMIZER_LR)

        # Experience Replay memory
        self.memory = PriorityReplayBuffer(buffer_size=BUFFER_SIZE, alpha=0.6, beta=0.4)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def act(self, state, eps=0.0):
        # Epsilon selection
        if random.random() < eps:
            return random.choice(np.arange(self.action_size))

        # Compute action values
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Greedy action selection
        return np.argmax(action_values.cpu().data.numpy())

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones, weights, idx = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        error = (Q_targets - Q_expected).squeeze().pow(2) * weights
        loss = error.mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) 

        #-------------------- update experience priority -------------- #
        with torch.no_grad():
            experience_loss = (Q_targets - Q_expected).detach().squeeze().abs()
            self.memory.update_priorities(idx, experience_loss)

    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def step_error(self, state, action, reward, next_state, done):
        # To tensor
        state = torch.from_numpy(state).float().unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)

        self.qnetwork_local.eval()
        with torch.no_grad():
            next_state_value = self.qnetwork_local(next_state)
            next_state_value = np.argmax(next_state_value.cpu().data.numpy()[0])

            action_values = self.qnetwork_local(state)
            action_values = action_values.cpu().data.numpy()[0]
        self.qnetwork_local.train() 

        return reward + GAMMA * next_state_value * (1-done) - action_values[action]