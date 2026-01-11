"""MAPPO Policy Networks - Actor and Critic."""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from typing import Tuple, Dict
import numpy as np


class Actor(nn.Module):
    """Actor network with discrete and continuous action heads."""
    
    def __init__(self, obs_dim: int, discrete_dim: int = 5, continuous_dim: int = 4,
                 hidden_dims: Tuple[int, ...] = (256, 128, 64)):
        super().__init__()
        
        # Backbone
        layers = []
        in_dim = obs_dim
        for h in hidden_dims[:-1]:
            layers.extend([nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()])
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        
        # Discrete head
        self.discrete_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]), nn.LayerNorm(hidden_dims[-1]), nn.ReLU(),
            nn.Linear(hidden_dims[-1], discrete_dim),
        )
        
        # Continuous head
        self.continuous_mean = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]), nn.LayerNorm(hidden_dims[-1]), nn.ReLU(),
            nn.Linear(hidden_dims[-1], continuous_dim), nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(continuous_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor):
        features = self.backbone(obs)
        discrete_logits = self.discrete_head(features)
        cont_mean = self.continuous_mean(features)
        cont_std = torch.exp(self.log_std).expand_as(cont_mean)
        return discrete_logits, cont_mean, cont_std
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        logits, mean, std = self.forward(obs)
        
        disc_dist = Categorical(logits=logits)
        disc_action = logits.argmax(-1) if deterministic else disc_dist.sample()
        
        cont_dist = Normal(mean, std)
        cont_action = mean if deterministic else cont_dist.sample()
        cont_action = torch.clamp(cont_action, -1, 1)
        
        log_prob = disc_dist.log_prob(disc_action) + cont_dist.log_prob(cont_action).sum(-1)
        
        return {"discrete": disc_action, "continuous": cont_action}, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, disc_action: torch.Tensor, cont_action: torch.Tensor):
        logits, mean, std = self.forward(obs)
        
        disc_dist = Categorical(logits=logits)
        cont_dist = Normal(mean, std)
        
        log_prob = disc_dist.log_prob(disc_action) + cont_dist.log_prob(cont_action).sum(-1)
        entropy = disc_dist.entropy() + cont_dist.entropy().sum(-1)
        
        return log_prob, entropy


class Critic(nn.Module):
    """Centralized Critic for MAPPO."""
    
    def __init__(self, state_dim: int, hidden_dims: Tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class MAPPOPolicy:
    """MAPPO Policy wrapper."""
    
    def __init__(self, obs_dim: int, state_dim: int, discrete_dim: int = 5, continuous_dim: int = 4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        
        self.actor = Actor(obs_dim, discrete_dim, continuous_dim).to(device)
        self.critic = Critic(state_dim).to(device)
    
    def act(self, obs: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob = self.actor.get_action(obs_t, deterministic)
            return {
                "discrete": action["discrete"].cpu().numpy()[0],
                "continuous": action["continuous"].cpu().numpy()[0],
            }, log_prob.cpu().numpy()[0]
    
    def get_value(self, state: np.ndarray) -> float:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.critic(state_t).cpu().numpy()[0, 0]
    
    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
