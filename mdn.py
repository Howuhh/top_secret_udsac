import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from torch.distributions import Categorical, Normal

LOG_STD_MIN = -2
LOG_STD_MAX = 20 # default 20


class MDN(nn.Module):
    def __init__(self, input_size, output_size, n_heads):
        super().__init__()
        
        hidden_size = 64

        self.output_size = output_size
        self.n_heads = n_heads
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.log_alpha_layer = nn.Sequential(
            nn.Linear(hidden_size, n_heads),
            nn.LogSoftmax(dim=-1)
        )
        self.log_sigma_layer = nn.Linear(hidden_size, output_size * n_heads)
        self.mu_layer = nn.Linear(hidden_size, output_size * n_heads)
        
    def forward(self, x):
        hidden = self.model(x)
    
        log_alpha = self.log_alpha_layer(hidden)
        mu = self.mu_layer(hidden).reshape(-1, self.n_heads, self.output_size)
        log_sigma = self.log_sigma_layer(hidden).reshape(-1, self.n_heads, self.output_size)
        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
       
        return log_alpha, mu, log_sigma.exp()
            
    def log_prob(self, log_alpha, mu, sigma, y):
        mixture_dist = Normal(mu, sigma)
        
        # (batch, output_size) -> (batch, 1, output_size) -> (batch, n_heads, output_size)
        y = y.unsqueeze(-2).expand(-1, self.n_heads, -1)
        
        return torch.logsumexp(mixture_dist.log_prob(y).sum(-1) + log_alpha, dim=-1)

    def sample(self, log_alpha, mu, sigma):        
        alpha_dist = Categorical(probs=log_alpha.exp())
        mixture_dist = Normal(mu, sigma)
        
        # just selecting mixture_idx from mixture samples for each input in batch
        mixture_idx = alpha_dist.sample().view(-1, 1, 1).expand(-1, -1, self.output_size)
        mixture_sample = mixture_dist.sample()

        return mixture_sample.gather(-2, mixture_idx).view(-1, self.output_size)
    
    def mean(self, log_alpha, mu, sigma):
        mixture_dist = Normal(mu, sigma)
        
        return (mixture_dist.mean * log_alpha.exp().unsqueeze(-1)).sum(-2)
    
    def nll_loss(self, log_alpha, mu, sigma, y):
        return -self.log_prob(log_alpha, mu, sigma, y).mean()