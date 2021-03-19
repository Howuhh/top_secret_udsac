import torch
import torch.nn as nn
import numpy as np

from torch.distributions import Categorical, Normal


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
        self.alpha_layer = nn.Sequential(
            nn.Linear(hidden_size, n_heads),
            nn.Softmax(dim=-1)
        )
        self.log_sigma_layer = nn.Linear(hidden_size, output_size * n_heads)
        self.mu_layer = nn.Linear(hidden_size, output_size * n_heads)
        
    def forward(self, x):
        hidden = self.model(x)
    
        alpha = self.alpha_layer(hidden)
        log_sigma = self.log_sigma_layer(hidden).reshape(-1, self.n_heads, self.output_size)
        mu = self.mu_layer(hidden).reshape(-1, self.n_heads, self.output_size)
                
        return alpha, mu, log_sigma.exp()
    
    def prob(self, alpha, mu, sigma, y):
        mixture_dist = Normal(mu, sigma)
        
        # (batch, output_size) -> (batch, 1, output_size) -> (batch, n_heads, output_size)
        y = y.unsqueeze(-2).expand(-1, self.n_heads, -1)
        
        return torch.sum(mixture_dist.log_prob(y).sum(-1).exp() * alpha, dim=-1)
        
    def log_prob(self, alpha, mu, sigma, y):
        mixture_dist = Normal(mu, sigma)
        
        y = y.unsqueeze(-2).expand(-1, self.n_heads, -1)
        
        return torch.logsumexp(mixture_dist.log_prob(y).sum(-1) + torch.log(alpha), dim=-1)

    def sample(self, alpha, mu, sigma):        
        alpha_dist = Categorical(probs=alpha)
        mixture_dist = Normal(mu, sigma)
        
        # just selecting mixture_idx from mixture samples for each input in batch
        mixture_idx = alpha_dist.sample().view(-1, 1, 1).expand(-1, -1, self.output_size)
        mixture_sample = mixture_dist.sample()

        return mixture_sample.gather(-2, mixture_idx).view(-1, self.output_size)
    
    def mean(self, alpha, mu, sigma):
        mixture_dist = Normal(mu, sigma)
        
        return (mixture_dist.mean * alpha.unsqueeze(-1)).sum(-2)    
    
    def nll_loss(self, x, y):
        alpha, mu, sigma = self.forward(x)
        
        log_prob = self.log_prob(alpha, mu, sigma, y)
        
        return -log_prob.mean()
