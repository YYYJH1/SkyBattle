"""Rollout Buffer for MAPPO."""

import numpy as np
import torch
from typing import Generator


class RolloutBuffer:
    """Rollout buffer for storing trajectories and computing GAE."""
    
    def __init__(self, buffer_size: int, obs_dim: int, state_dim: int, n_agents: int,
                 continuous_dim: int = 4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.continuous_dim = continuous_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.reset()
    
    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_agents, self.obs_dim), dtype=np.float32)
        self.states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.discrete_actions = np.zeros((self.buffer_size, self.n_agents), dtype=np.int64)
        self.continuous_actions = np.zeros((self.buffer_size, self.n_agents, self.continuous_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.ptr = 0
        self.full = False
    
    def add(self, observations, state, actions, rewards, dones, log_probs, values, agent_ids):
        for i, aid in enumerate(agent_ids):
            self.observations[self.ptr, i] = observations[aid]
            self.discrete_actions[self.ptr, i] = actions[aid]["discrete"]
            self.continuous_actions[self.ptr, i] = actions[aid]["continuous"]
            self.rewards[self.ptr, i] = rewards[aid]
            self.dones[self.ptr, i] = float(dones[aid])
            self.log_probs[self.ptr, i] = log_probs[aid]
            self.values[self.ptr, i] = values[aid]
        self.states[self.ptr] = state
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
    
    def compute_advantages(self, last_values: np.ndarray):
        last_gae = np.zeros(self.n_agents)
        for t in reversed(range(self.ptr)):
            next_values = last_values if t == self.ptr - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]
    
    def get(self, batch_size: int, shuffle: bool = True) -> Generator:
        size = self.ptr * self.n_agents
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        
        obs_flat = self.observations[:self.ptr].reshape(-1, self.obs_dim)
        disc_flat = self.discrete_actions[:self.ptr].reshape(-1)
        cont_flat = self.continuous_actions[:self.ptr].reshape(-1, self.continuous_dim)
        log_probs_flat = self.log_probs[:self.ptr].reshape(-1)
        adv_flat = self.advantages[:self.ptr].reshape(-1)
        ret_flat = self.returns[:self.ptr].reshape(-1)
        states_flat = np.repeat(self.states[:self.ptr], self.n_agents, axis=0)
        
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        
        for start in range(0, size, batch_size):
            idx = indices[start:min(start + batch_size, size)]
            yield {
                "observations": torch.FloatTensor(obs_flat[idx]).to(self.device),
                "states": torch.FloatTensor(states_flat[idx]).to(self.device),
                "discrete_actions": torch.LongTensor(disc_flat[idx]).to(self.device),
                "continuous_actions": torch.FloatTensor(cont_flat[idx]).to(self.device),
                "old_log_probs": torch.FloatTensor(log_probs_flat[idx]).to(self.device),
                "advantages": torch.FloatTensor(adv_flat[idx]).to(self.device),
                "returns": torch.FloatTensor(ret_flat[idx]).to(self.device),
            }
    
    def is_full(self) -> bool:
        return self.full or self.ptr >= self.buffer_size
