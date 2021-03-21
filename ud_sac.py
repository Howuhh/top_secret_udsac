import os
import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from copy import deepcopy
from itertools import chain

from core import Actor, Critic, RandomController
from core import ReplayBuffer, Episode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(env, seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    

class UDSAC:
    def __init__(self, state_size, action_size, command_scale=(1, 1), critic_heads=10, alpha=0.2, gamma=0.99, tau=0.005, actor_lr=1e-4, critic_lr=1e-4):
        self.actor = Actor(state_size, action_size, command_scale).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr) 
        
        # PROBLEM: action_size != num_actions, num_actions=4 but action_size=1
        self.critic = Critic(state_size, action_size, command_size=2, n_heads=critic_heads).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.action_size = action_size    
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
                    
    def _actor_loss(self, state, command):
        _, action_probs, action_log_probs = self.actor(state, command, return_probs=True)
        
        Q_target = self.critic.log_prob_by_aciton(state, command, output=command).exp()

        assert action_log_probs.shape == Q_target.shape == action_probs.shape
        
        loss = ((self.alpha * action_log_probs - Q_target.detach()) * action_probs).sum(dim=1).mean()
        
        return loss
    
    def _critic_loss(self, state, command, action, reward, next_state, done, output):
        with torch.no_grad():
            next_command = torch.zeros_like(command)

            next_command[:, 0] = command[:, 0] - reward
            next_command[:, 1] = torch.max(command[:, 1] - 1, torch.ones_like(command[:, 1]))
            
            next_action = self.actor(next_state, next_command)
            next_action = F.one_hot(next_action.long(), num_classes=self.action_size)
            
            next_output = self.critic.sample(next_state, next_action, next_command)
            target_output = next_output + torch.cat([reward.view(-1, 1), torch.ones_like(reward).view(-1, 1)], dim=-1)
        
        Q_done = self.critic.log_prob(state, action, command, output)
        Q_not_done = self.critic.log_prob(state, action, command, target_output)        
        loss = -(done * Q_done + (1 - done) * Q_not_done).mean()

        return loss
    
    def update(self, batch):
        state, command, action, reward, next_state, done, output = batch
        
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        command = torch.tensor(command, dtype=torch.float32, device=DEVICE)
        
        action = torch.tensor(action, dtype=torch.float32, device=DEVICE)
        action = F.one_hot(action.long(), num_classes=self.action_size)
        
        reward = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float32, device=DEVICE)
        output = torch.tensor(output, dtype=torch.float32, device=DEVICE)
        
        actor_loss = self._actor_loss(state, command)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        print("Actor: ", [torch.norm(p.grad) for p in self.actor.parameters()])
        
        critic_loss = self._critic_loss(state, command, action, reward, next_state, done, output)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        print("Critic: ", [torch.norm(p.grad) for p in self.critic.parameters()])

    def act(self, state, command, eval_mode=False):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            command = torch.tensor(command, dtype=torch.float32, device=DEVICE) 

            action = self.actor(state, command, eval_mode=eval_mode)
            
        return action.cpu().numpy().item()
    
    
def sample_episode(env, agent, controller, eval_mode=False, seed=0):
    states, actions, rewards, commands, next_states, dones = [], [], [], [], [], []
        
    if eval_mode:
        set_seed(env, seed)
    
    state, done = env.reset(), False
    desired_return, desired_horizon = (1, 1) if controller is None else controller.get_command(state)
    
    total_reward, length = 0.0, 0.0
    
    while not done:
        command = np.array([desired_return, desired_horizon])
        
        action = env.action_space.sample() if agent is None else agent.act(state, command, eval_mode=eval_mode)
        
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        commands.append(command)
        next_states.append(next_state)
        dones.append(done)
        
        state = next_state
        
        desired_return = min(desired_return - reward, env.reward_range[1])
        desired_horizon = max(desired_horizon - 1, 1)
        
        total_reward += reward
        length += 1
    
    return Episode(
        states=np.array(states),
        actions=np.array(actions),
        rewards=np.array(rewards),
        commands=np.array(commands),
        next_states=np.array(next_states), 
        dones=np.array(dones),
        total_return=total_reward,
        length=length
    )


def train(env_name, agent, controller, warmup_episodes=10, iterations=700, episodes_per_iter=20, updates_per_iter=100, 
          buffer_size=1000, batch_size=128, test_episodes=10, test_every=5, seed=0):
    print("training on", DEVICE)
    
    env, test_env = gym.make(env_name), gym.make(env_name)
    set_seed(env, seed=seed)
    
    buffer = ReplayBuffer(size=buffer_size)
    
    print("Start WarmUp")
    for _ in range(warmup_episodes):
        episode = sample_episode(env, None, None)
        buffer.add(episode)
      
    print("Start Training")  
    for i in range(iterations):
        for _ in range(updates_per_iter):
            batch = buffer.sample(batch_size)
            
            states, actions, rewards, commands, next_states, dones, outputs = [], [], [], [], [], [], []

            for episode in batch:
                T = episode.length

                t1 = np.random.randint(0, T)
                t2 = int(T) # take only full run
                dr = np.sum(episode.rewards[t1:t2])
                dh = t2-t1

                states.append(episode.states[t1])
                actions.append(episode.actions[t1])
                rewards.append(episode.rewards[t1])
                commands.append(episode.commands[t1])
                next_states.append(episode.next_states[t1])
                dones.append(episode.dones[t1])
                outputs.append([dr, dh])

            agent.update([states, commands, actions, rewards, next_states, dones, outputs])
        
        for _ in range(episodes_per_iter):
            episode = sample_episode(env, agent, controller)
            buffer.add(episode)
            
        if i % test_every == 0:
            eval_episodes = [sample_episode(test_env, agent, controller) for _ in range(test_episodes)]

            returns = np.array([e.total_return for e in eval_episodes])
            desired_returns = np.array([e.commands[0][0] for e in eval_episodes])
            desired_horizons = np.array([e.commands[0][1] for e in eval_episodes])
            
            print(i, f"Reward: {returns.mean()} ({returns.std()})", f"Desired: {desired_returns.mean()} ({desired_returns.std()})")
        
    
if __name__ == "__main__":
    agent = UDSAC(8, 4, actor_lr=1e-3, critic_lr=1e-3)
    controller = RandomController(-200, 200)
    
    # with torch.autograd.detect_anomaly():
    train("LunarLander-v2", agent, controller)