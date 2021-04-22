import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        hidden_layers = [128, 64]
        
        self.fc1 = nn.Linear(state_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.V = nn.Linear(hidden_layers[1], 1)
        self.A = nn.Linear(hidden_layers[1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        value = self.V(x)
        advantage = self.A(x)
        
        return advantage.sub_(advantage.mean()).add_(value)
