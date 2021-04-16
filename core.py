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
    dones: np.ndarray
    total_return: float
    length: int
    
    
class ReplayBuffer():
    def __init__(self):
        self.buffer = []

    def add_episodes(self, episodes):
        self.buffer = self._cut_episodes(episodes)
    
    def _cut_episodes(self, episodes):
        fragments = [] # (state, command, action, reward, output)
        
        for episode in episodes:
            prefix_return = 0.0
            for t in range(episode.length):
                state = episode.states[t]
                action = episode.actions[t]
                reward = episode.rewards[t]
                command = episode.commands[t]
                output = [episode.total_return - prefix_return, episode.length - t]
                
                prefix_return += reward
                
                fragments.append([state, command, action, reward, output])
            
        return fragments
    
    def clear(self):
        self.buffer = []

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self), batch_size)
        
        return list(zip(*[self.buffer[idx] for idx in idxs]))

    def __len__(self):
        return len(self.buffer)


class CartPolev0RandomController:
    def __init__(self, low, high):  
        self.low = low
        self.high = high

    def get_command(self, state):
        desired_return = np.round(np.random.uniform(self.low, self.high))
        
        return desired_return, desired_return
    
    def update(self, state):
        return 0
    
    def save(self, path):
        pass
    

class RandomController:
    def __init__(self, desired_return_range, desired_horizon_range):  
        self.r_low, self.r_high = desired_return_range
        self.h_low, self.h_high = desired_horizon_range

    def get_command(self, state):
        desired_return = np.round(np.random.uniform(self.r_low, self.r_high))
        desired_horizon = np.round(np.random.uniform(self.h_low, self.h_high))
        
        return desired_return, np.round(desired_horizon)
    
    def update(self, state):
        return 0
    
    def save(self, path):
        pass
    
        
class MDNController:
    def __init__(self, state_size, command_size, mdn_heads, lr, sigma_scale=1.0):  
        self.model = MDN(state_size, command_size, n_heads=mdn_heads, clip=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.sigma_scale = sigma_scale

    def get_command(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        
        log_alpha, mu, sigma = self.model(state)
        sigma = self.sigma_scale * sigma
        
        command = self.model.sample(log_alpha, mu, sigma).squeeze().cpu().numpy()
        
        return command[0], np.maximum(command[1], 1)
        
    def update(self, batch):
        state, _, _, _, output = batch

        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        output = torch.tensor(output, dtype=torch.float32, device=DEVICE)

        log_alpha, mu, sigma = self.model(state)
        sigma = self.sigma_scale * sigma
        
        loss = self.model.nll_loss(log_alpha, mu, sigma, output)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def save(self, path):
        torch.save(self, path)
        

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
        self.model = MDN(state_size + command_size + action_size, command_size, n_heads, clip=True)
        
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
        