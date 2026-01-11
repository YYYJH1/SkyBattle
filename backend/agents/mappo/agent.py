"""MAPPO Agent - Training and Inference."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path

from .policy import MAPPOPolicy
from .buffer import RolloutBuffer


class MAPPOAgent:
    """Multi-Agent PPO Agent with CTDE paradigm."""
    
    def __init__(self, obs_dim: int, state_dim: int, n_agents: int,
                 discrete_dim: int = 5, continuous_dim: int = 4,
                 lr_actor: float = 3e-4, lr_critic: float = 5e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5,
                 n_epochs: int = 10, batch_size: int = 256, buffer_size: int = 2048,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.device = device
        
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.policy = MAPPOPolicy(obs_dim, state_dim, discrete_dim, continuous_dim, device)
        
        self.actor_optimizer = optim.Adam(self.policy.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.policy.critic.parameters(), lr=lr_critic)
        
        self.buffer = RolloutBuffer(buffer_size, obs_dim, state_dim, n_agents,
                                    continuous_dim, gamma, gae_lambda, device)
        
        self.total_steps = 0
        self.total_episodes = 0
    
    def act(self, observations: Dict[str, np.ndarray], deterministic: bool = False):
        actions, log_probs = {}, {}
        for agent_id, obs in observations.items():
            action, lp = self.policy.act(obs, deterministic)
            actions[agent_id] = {"discrete": int(action["discrete"]), "continuous": action["continuous"].tolist()}
            log_probs[agent_id] = float(lp)
        return actions, log_probs
    
    def get_values(self, state: np.ndarray, agent_ids: List[str]) -> Dict[str, float]:
        value = self.policy.get_value(state)
        return {aid: value for aid in agent_ids}
    
    def store_transition(self, observations, state, actions, rewards, dones, log_probs, values, agent_ids):
        actions_np = {aid: {"discrete": np.array(a["discrete"]), "continuous": np.array(a["continuous"])}
                      for aid, a in actions.items()}
        self.buffer.add(observations, state, actions_np, rewards, dones, log_probs, values, agent_ids)
    
    def update(self, last_values: np.ndarray) -> Dict[str, float]:
        self.buffer.compute_advantages(last_values)
        
        total_actor_loss, total_critic_loss, total_entropy, n_updates = 0.0, 0.0, 0.0, 0
        
        for _ in range(self.n_epochs):
            for batch in self.buffer.get(self.batch_size):
                log_probs, entropy = self.policy.actor.evaluate_actions(
                    batch["observations"], batch["discrete_actions"], batch["continuous_actions"])
                values = self.policy.critic(batch["states"]).squeeze(-1)
                
                ratio = torch.exp(log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch["advantages"]
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                value_loss = nn.functional.mse_loss(values, batch["returns"])
                
                self.actor_optimizer.zero_grad()
                (actor_loss + self.entropy_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        self.buffer.reset()
        return {"actor_loss": total_actor_loss / n_updates, "critic_loss": total_critic_loss / n_updates,
                "entropy": total_entropy / n_updates}
    
    def should_update(self) -> bool:
        return self.buffer.is_full()
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy": {"actor": self.policy.actor.state_dict(), "critic": self.policy.critic.state_dict()},
            "actor_opt": self.actor_optimizer.state_dict(), "critic_opt": self.critic_optimizer.state_dict(),
            "total_steps": self.total_steps, "total_episodes": self.total_episodes,
        }, path)
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.actor.load_state_dict(ckpt["policy"]["actor"])
        self.policy.critic.load_state_dict(ckpt["policy"]["critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        self.critic_optimizer.load_state_dict(ckpt["critic_opt"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.total_episodes = ckpt.get("total_episodes", 0)
