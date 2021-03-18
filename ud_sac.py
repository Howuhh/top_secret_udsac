import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from copy import deepcopy
from itertools import chain
from core import Actor, Critic

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UDSAC:
    def __init__(self, state_size, action_size, command_scale=(1, 1), critic_heads=10, alpha=0.2, gamma=0.99, tau=0.005, actor_lr=1e-4, critic_lr=1e-4):
        self.actor = Actor(state_size, action_size, command_scale).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr) 
        
        self.critic = Critic(state_size, action_size, command_size=2, n_heads=critic_heads).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
            
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
                    
    def _actor_loss(self, state, command):
        action, action_probs, action_log_probs = self.actor(state, command, return_probs=True)
        
        Q_traget = self.critic.log_prob_by_aciton(state, command, output=command).exp()

        assert action_log_probs.shape == Q_target.shape == action_probs.shape
        
        loss = (self.alpha * action_log_probs - Q_target) * action_probs).sum(dim=1).mean()
        
        return loss
    
    def _critic_loss(self, state, command, action, reward, next_state, done, output):
        Q_done = self.critic.log_prob(state, action, command, output)

        next_command = torch.zeros_like(command)
        next_command[:, 0] = command[:, 0] - reward
        next_command[:, 1] = torch.min(command[:, 1] - 1, 1)
        
        next_action = self.actor(next_state, next_command)
        next_output = self.critic.sample(next_state, next_action, next_command)

        target_output = next_output + torch.tensor([reward, 1], dtype=torch.float32, device=DEVICE)
        
        Q_not_done = torch.min(
            self.critic1.log_prob(state, action, command, target_output),
            self.critic2.log_prob(state, action, command, target_output)
        )
        loss = -(done * Q_done + (1 - done) * Q_not_done).mean()

        return loss
    
    def update(self, batch):
        state, command, action, reward, next_state, done, output = batch
        
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        command = torch.tensor(command, dtype=torch.float32, device=DEVICE)
        action = torch.tensor(action, dtype=torch.float32, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float32, device=DEVICE)
        output = torch.tensor(output, dtype=torch.float32, device=DEVICE)
        
        actor_loss = self._actor_loss(state, command)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        critic_loss = self._critic_loss(state, command, action, reward, next_state, done, output)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def act(self, state, command, eval_mode=False):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            command = torch.tensor(command, dtype=torch.float32, device=DEVICE) 

            action = self.actor(state, command, eval_mode=eval_mode)
            
        return action.cpu().numpu().item()
    

# TODO
class ReplayBuffer:
    pass
    
def sample_episode():
    pass

def train():
    pass