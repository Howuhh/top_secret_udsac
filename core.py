import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from dataclasses import dataclass
from collections import deque
from torch.distributions import Categorical

from mdn import MDN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Episode:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    commands: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    total_return: float
    length: int


class ReplayBuffer():
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.size = size

    def add(self, episode):
        self.buffer.append(episode)

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self), batch_size)
        return [self.buffer[idx] for idx in idxs]

    def sort(self, cmp=lambda episode: episode.total_return):
        self.buffer = sorted(self.buffer, key=cmp)[-self.size:]

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class RandomController:
    def __init__(self, low, high):  
        self.low = low
        self.high = high

    def get_command(self, state):
        desired_return = np.round(np.random.uniform(self.low, self.high))
        
        return desired_return, desired_return
    
    def consume_episode(self, episode):
        pass

    def sort(self):
        pass


class NormalController:
    def __init__(self, buffer_size):  
        self.buffer = ReplayBuffer(buffer_size)

    def get_command(self, state):
        returns = [e.total_return for e in self.buffer.buffer]
        horizons = [e.length for e in self.buffer.buffer]
        
        returns_mean, returns_std = np.mean(returns), np.std(returns)
        
        desired_return = np.random.normal((returns_mean + (returns_mean + returns_std)) / 2, returns_std)
        desired_horizon = np.mean(horizons)
        
        return desired_return, np.round(desired_horizon)
    
    def consume_episode(self, episode):
        self.buffer.add(episode)

    def sort(self):
        self.buffer.sort()
    

class MeanController:
    def __init__(self, buffer_size, std_scale=1.0):  
        self.std_scale = std_scale   
        self.buffer = ReplayBuffer(buffer_size)

    def get_command(self, state):
        returns = [e.total_return for e in self.buffer.buffer]
        horizons = [e.length for e in self.buffer.buffer]
        
        returns_mean, returns_std = np.mean(returns), np.std(returns)
        
        desired_return = np.random.uniform(returns_mean, returns_mean + self.std_scale * returns_std)
        desired_horizon = np.mean(horizons)
        
        return desired_return, np.round(desired_horizon)
    
    def consume_episode(self, episode):
        self.buffer.add(episode)

    def sort(self):
        self.buffer.sort()


class Actor(nn.Module):
    def __init__(self, state_size, action_size, command_scale=(1, 1)):
        super().__init__()
        
        embedding_size = 64
        hidden_size = 256
        
        self.command_scale = torch.tensor(command_scale, dtype=torch.float32, device=DEVICE)
        
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
        
    def _get_logits(self, state, command):
        state_output = self.state_layer(state)

        command_output = self.command_layer(command * self.command_scale)

        return self.action_layer(state_output * command_output)
        
    def forward(self, state, command, eval_mode=False, return_probs=False):
        logits = self._get_logits(state, command)
              
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


class Critic(nn.Module):
    def __init__(self, state_size, action_size, command_size, n_heads=4):
        super().__init__()
        self.action_size = action_size
        # Q(s, a, c) = P(output == command | state, command, action)
        self.model = MDN(state_size + command_size + action_size, command_size, n_heads)
        
    def sample(self, state, command, action):            
        x = torch.cat([state, command, action], dim=1)
        log_alpha, mu, sigma = self.model(x)
        
        return self.model.sample(log_alpha, mu, sigma)
        
    def log_prob(self, state, command, action, output):
        x = torch.cat([state, command, action], dim=1)
        log_alpha, mu, sigma = self.model(x)
            
        return self.model.log_prob(log_alpha, mu, sigma, output)
    
    # only for discrete actions
    def log_prob_by_action(self, state, command, output):        
        log_probs = []
        
        for a in range(self.action_size):    
            action = torch.ones(state.shape[0], device=DEVICE) * a
            action = F.one_hot(action.long(), num_classes=self.action_size)

            los_prob = self.log_prob(state, command, action, output).view(-1, 1)
            log_probs.append(los_prob)
        
        return torch.cat(log_probs, dim=-1)
        
    def mean(self, state, command, action):
        x = torch.cat([state, command, action], dim=1)
        log_alpha, mu, sigma = self.model(x)
        
        return self.model.mean(log_alpha, mu, sigma)
    
    def nll_loss(self, state, command, action, output):
        x = torch.cat([state, command, action], dim=1)
        log_alpha, mu, sigma = self.model(x)
        
        return self.model.nll_loss(log_alpha, mu, sigma, output)
        