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
from collections import defaultdict

from joblib import delayed, Parallel

from core import Actor, Critic, RandomController, CartPolev0RandomController, MDNController
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
    def __init__(self, state_size, action_size, command_scale=(1, 1), critic_heads=10, init_alpha=None, target_entropy_scale=0.98, 
                 gamma=0.99, tau=0.005, actor_lr=1e-4, critic_lr=1e-4, alpha_lr=1e-4):
        self.actor = Actor(state_size, action_size, command_scale).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr) 
        
        self.critic = Critic(state_size, action_size, command_size=2, n_heads=critic_heads).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.target_critic = deepcopy(self.critic)
        
        self.action_size = action_size    
        self.gamma = gamma
        self.tau = tau
        
        self.init_alpha = 0.0 if init_alpha is None else np.log(init_alpha)
        # max possible entropy (from paper)
        self.target_entropy = -np.log((1.0 / action_size)) * target_entropy_scale
  
        self.log_alpha = torch.tensor([self.init_alpha], dtype=torch.float32, device=DEVICE, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp()
        
    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)
                    
    def _actor_loss(self, state, command):
        _, action_probs, action_log_probs = self.actor(state, command, return_probs=True)
        
        Q_target = self.target_critic.log_prob_by_action(state, command, output=command) # .exp()
        # Q_target = self.critic.log_prob_by_action(state, command, output=command) # .exp()
        
        assert action_log_probs.shape == Q_target.shape == action_probs.shape
        
        loss = ((self.alpha * action_log_probs - Q_target.detach()) * action_probs).sum(dim=1).mean()
        
        return loss

    def _alpha_loss(self, state, command):
        with torch.no_grad():
            action, action_probs, action_log_probs = self.actor(state, command, return_probs=True)
            # https://github.com/yining043/SAC-discrete/issues/2#event-3685116634
            action_log_probs_exp = (action_log_probs * action_probs).sum(dim=1)

        loss = (-self.log_alpha * (action_log_probs_exp + self.target_entropy)).mean()

        return loss
    
    def _critic_loss_nll(self, state, command, action, output):
        return self.critic.nll_loss(state, command, action, output)

    def update(self, batch):        
        state, command, action, reward, output = batch
        
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        command = torch.tensor(command, dtype=torch.float32, device=DEVICE)
        
        action = torch.tensor(action, dtype=torch.float32, device=DEVICE)
        action = F.one_hot(action.long(), num_classes=self.action_size)
        
        reward = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
        output = torch.tensor(output, dtype=torch.float32, device=DEVICE)
        
        critic_loss = self._critic_loss_nll(state, command, action, output)
    
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = self._actor_loss(state, command) + self._actor_loss(state, output)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        alpha_loss = self._alpha_loss(state, command)
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        with torch.no_grad():
            self._soft_update(self.target_critic, self.critic)
            
        return actor_loss, critic_loss


    def act(self, state, command, eval_mode=False):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            command = torch.tensor(command, dtype=torch.float32, device=DEVICE) 

            action = self.actor(state, command, eval_mode=eval_mode)
            
        return action.cpu().numpy().item()
    
    def save(self, path):
        torch.save(self, path)


def rollout(env, agent, desired_return, desired_horizon):
    state, done = env.reset(), False

    total_return, steps = 0.0, 0.0 
    while not done:
        command = np.array([desired_return, desired_horizon])
        action = agent.act(state, command, eval_mode=True)
        state, reward, done, _ = env.step(action)
        
        desired_return = min(desired_return - reward, env.reward_range[1])
        desired_horizon = max(desired_horizon - 1, 1)   

        steps += 1
        total_return += reward

    return total_return, steps


def evaluate_agent(env_name, agent, desired_return_range, desired_horizon_range, num=50, seed=42):
    return_grid = np.linspace(desired_return_range[0], desired_return_range[1], num)
    horizon_grid = np.linspace(desired_horizon_range[0], desired_horizon_range[1], num)
    
    env = gym.make(env_name)
    set_seed(env, seed=seed) # мб на одной энве всегда тестировать?

    actual_desired_error = np.zeros((num, num))
    
    for i, desired_reward in enumerate(return_grid):
        for j, desired_horizon in enumerate(horizon_grid):
            actual_reward, actual_horizon = rollout(env, agent, desired_reward, desired_horizon)
            
            actual_desired_error[i, j] = np.abs(desired_reward - actual_reward) + np.abs(desired_horizon - actual_horizon)

    return return_grid, horizon_grid, actual_desired_error


# def evaluate_agent(env_name, agent, desired_return_range, desired_horizon_range, num=50, seed=42):
#     return_grid = np.linspace(desired_return_range[0], desired_return_range[1], num)
#     horizon_grid = np.linspace(desired_horizon_range[0], desired_horizon_range[1], num)
    
#     env = gym.make(env_name)
#     set_seed(env, seed=seed) 

#     actual_return = np.array([rollout(env, agent, dr, dr)[0] for dr in return_grid])

#     return return_grid, horizon_grid, np.abs(return_grid - actual_return)
    


def sample_episode(env, agent, controller, eval_mode=False, seed=0):
    states, actions, rewards, commands, dones = [], [], [], [], []
            
    state, done = env.reset(), False
    desired_return, desired_horizon = (1, 1) if controller is None else controller.get_command(state)
    
    total_reward, length = 0.0, 0
    
    while not done:
        command = np.array([desired_return, desired_horizon])
        
        action = env.action_space.sample() if agent is None else agent.act(state, command, eval_mode=eval_mode)
        
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        commands.append(command)
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
        dones=np.array(dones),
        total_return=total_reward,
        length=length
    )


def train(env_name, agent, controller, eval_return_range, eval_horizon_range, warmup_episodes=10, iterations=700, episodes_per_iter=32, 
          updates_per_iter=100, batch_size=256, test_episodes=10, test_every=5, seed=0, partial_fit=False, n_jobs=4):
    print("training on", DEVICE)
    
    env, test_env = gym.make(env_name), gym.make(env_name)
    set_seed(env, seed=seed)
    
    buffer = ReplayBuffer()
    
    log = defaultdict(list)
    
    print("Start WarmUp")
    if partial_fit:
        pool = [delayed(sample_episode)(env, agent, controller) for _ in range(episodes_per_iter)]
    else:
        pool = [delayed(sample_episode)(env, None, None) for _ in range(warmup_episodes)]
    
    episodes = Parallel(n_jobs=n_jobs)(pool)
    buffer.add_episodes(episodes)
    
    total_critic_loss, total_actor_loss = 0.0, 0.0
    # total_controller_loss = 0.0
    
    best_eval_loss = np.inf
    print("Start Training")  
    for i in range(iterations):
        for _ in range(updates_per_iter):
            batch = buffer.sample(batch_size)
            
            actor_loss, critic_loss = agent.update(batch)
            
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            
        # for _ in range(updates_per_iter // 2):
        #     controller_loss = controller.update(batch)
        #     total_controller_loss += controller_loss
        
        pool = [delayed(sample_episode)(env, agent, controller) for _ in range(episodes_per_iter)]
        episodes = Parallel(n_jobs=n_jobs)(pool)
        buffer.add_episodes(episodes)
            
        if i % test_every == 0:
            _, _, error_matrix = evaluate_agent(env_name, agent, eval_return_range, eval_horizon_range, num=15, seed=seed)
            
            eval_loss = np.mean(error_matrix)
            eval_loss_std = np.std(error_matrix)
            
            print(f"Step: {i}, Eval loss: {round(eval_loss, 2)}", end=", ")
            
            actor_loss_mean = round(total_actor_loss.item() / ((i + 1) * updates_per_iter), 4)
            critic_loss_mean = round(total_critic_loss.item() / ((i + 1) * updates_per_iter), 4)
            # controller_loss_mean = round(total_controller_loss.item() / ((i + 1) * (updates_per_iter // 2)), 4)

            print(f"Actor loss: {actor_loss_mean}, Critic loss: {critic_loss_mean}, Alpha: {round(agent.alpha.detach().item(), 4)}")
            # print(f"Actor loss: {actor_loss_mean}, Critic loss: {critic_loss_mean}, Alpha: {round(agent.alpha.detach().item(), 4)}", end=", ")
            # print(f"Controller loss: {controller_loss_mean}")

            log["eval_loss_mean"].append(eval_loss)
            log["eval_loss_std"].append(eval_loss_std)
            log["actor_loss_mean"].append(actor_loss_mean)
            log["critic_loss_mean"].append(critic_loss_mean)
            # log["controller_loss_mean"].append(controller_loss_mean)
            
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                agent.save("udsac_agent_best.pt")
            
            agent.save("udsac_agent.pt")
            # controller.save("udsac_controller.pt")
            
    return log


if __name__ == "__main__":
    import pickle
    
    # agent = UDSAC(8, 4, actor_lr=1e-4, critic_lr=1e-4, critic_heads=5, target_entropy_scale=0.9, alpha_lr=1e-4, tau=0.01)
    # controller = RandomController((-300, 300), (50, 400))
    # controller = MDNController(8, 2, mdn_heads=5, lr=1e-4, sigma_scale=4.5)

    # log = train("LunarLander-v2", agent, controller, eval_return_range=(-300, 300), eval_horizon_range=(50, 300), warmup_episodes=10, 
                # iterations=10_000, episodes_per_iter=64, updates_per_iter=50, batch_size=1024, test_every=25, seed=42)
    
    # Идеи: реже обновлять контроллера (переобучается), ставить больше константу, ставить меньше энтропию
    
    agent = UDSAC(4, 2, actor_lr=1e-4, critic_lr=3e-4, critic_heads=5, target_entropy_scale=0.5, alpha_lr=1e-4, tau=0.001)
    controller = CartPolev0RandomController(low=10, high=195)
    # controller = MDNController(4, 2, mdn_heads=5, lr=1e-4, sigma_scale=5.0)

    log = train("CartPole-v0", agent, controller, eval_return_range=(10, 195), eval_horizon_range=(10, 195), warmup_episodes=10, 
                iterations=10_000, episodes_per_iter=256, updates_per_iter=100, batch_size=256, test_every=5, seed=42)
    
    with open("logs.pkl", "wb") as log_file:
        pickle.dump(log, log_file)