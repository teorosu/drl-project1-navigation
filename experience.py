from typing import Tuple
from numpy.core.fromnumeric import nonzero

from numpy.ma.core import where

import numpy as np
import torch

class PriorityReplayBuffer:
    def __init__(self, buffer_size:int, alpha, beta, nonzero_offset=0.001):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.nonzero_offset = nonzero_offset
        self.size = 0
        self.idx = 0

        self.memory = np.empty(buffer_size, dtype=[
            ('state', np.ndarray),
            ('action', np.int),
            ('reward', np.float),
            ('next_state', np.ndarray),
            ('done', np.bool),
            ('error', np.float)])

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        
        # Fetch next free entry and increment index
        entry = self.memory[self.idx]
        self.idx = (self.idx + 1) % self.buffer_size
        self.size = min(self.buffer_size, self.size+1)

        # Store entry details
        entry['state'] = state
        entry['action'] = action
        entry['reward'] = reward
        entry['next_state'] = next_state
        entry['done'] = done
        entry['error'] = np.max(self.memory['error'], initial=1.0)
    
    def _get_probabilities(self, alpha=1.0):
        scaled_probs = self.memory[0:self.size]
        scaled_probs = np.absolute(scaled_probs['error']) + self.nonzero_offset
        scaled_probs = scaled_probs ** alpha
        scaled_probs = scaled_probs / np.sum(scaled_probs)
        return scaled_probs

    def _get_weights(self, probabilities):
        """
        Importance weights
        w = ((N * P) ^ -β) / max(w)
        """
        weights = np.zeros(self.size)
        np.multiply(probabilities, self.size, out=weights)
        np.power(weights, -self.beta, out=weights, where=(weights!=0))
        np.divide(weights, weights.max(), out=weights)
        return weights

    def sample(self, batch_size) -> Tuple:
        # Compute scaled probabilities and importance weights
        probs = self._get_probabilities(self.alpha)
        weights = self._get_weights(probs)
        
        # Sample experiences
        idx = np.random.choice(self.size, size=batch_size, p=probs, replace=False)
        experiences = self.memory[idx]
        weights = weights[idx]

        # Generate tensors
        states = torch.from_numpy(np.vstack(experiences['state'])).float()
        actions = torch.from_numpy(np.vstack(experiences['action'])).long()
        rewards = torch.from_numpy(np.vstack(experiences['reward'])).float()
        next_states = torch.from_numpy(np.vstack(experiences['next_state'])).float()
        dones = torch.from_numpy(np.vstack(experiences['done']).astype(np.uint8)).float()
        weights = torch.from_numpy(np.vstack(weights)).float()

        # Update beta (anneal to 1)
        self.beta = 1 - self.beta * 0.995

        return (states, actions, rewards, next_states, dones, weights, idx)

    def update_priorities(self, idx: np.ndarray, errors: torch.tensor):
        errors = errors.numpy()
        self.memory[idx]['error'] = errors

    def __len__(self):
        return self.size
