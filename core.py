import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.distributions import Categorical


class RandomController:
    def __init__(self, low=-200, high=200):        
        self.low = low
        self.high = high
        
    def get_command(self, state):
        desired_return = np.random.uniform(self.low, self.high)
        desired_horizon = 90
        
        return desired_return, desired_horizon


class Actor(nn.Module):
    def __init__(self, state_size, action_size, command_scale=(1, 1)):
        super().__init__()
        
        embedding_size = 64
        hidden_size = 256
        
        self.command_scale = torch.tensor(command_scale, dtype=torch.float32)
        
        self.command_layer = nn.Sequential(
            nn.Linear(2, embedding_size),
            nn.Sigmoid()
        )
        self.state_layer = nn.Sequential(
            nn.Linear(state_size, embedding_size),
            nn.Tanh() # Правда ли он тут нужен?
        )
        self.action_layer = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
    def get_logits(self, state, command):
        state_output = self.state_layer(state)
        command_output = self.command_layer(command * self.command_scale)
        
        return self.action_layer(state_output * command_output)
        
    def forward(self, state, command, eval_mode=False, return_probs=False):
        logits = self.get_logits(state, command)
        
        probs = F.softmax(logits, dim=-1)
        policy_dist = Categorical(probs=probs)

        if eval_mode:
            action = torch.argmax(probs, dim=-1)
        else:
            action = policy_dist.sample()

        if return_probs:
            log_probs = F.log_softmax(logits, dim=-1)

            return action, probs, log_probs

        return action
    
class Crititc(nn.Module):
    def __init__(self, state_size, command_size):
        super().__init__()
        pass
    