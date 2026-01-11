"""Tests for the MAPPO Agent."""

import pytest
import numpy as np
import torch
from backend.agents.mappo import MAPPOAgent, MAPPOPolicy, Actor, Critic, RolloutBuffer


class TestActor:
    """Tests for the Actor network."""
    
    def test_actor_creation(self):
        """Test actor initialization."""
        actor = Actor(obs_dim=65, discrete_dim=5, continuous_dim=4)
        
        assert actor.obs_dim == 65
        assert actor.discrete_action_dim == 5
        assert actor.continuous_action_dim == 4
    
    def test_actor_forward(self):
        """Test actor forward pass."""
        actor = Actor(obs_dim=65)
        obs = torch.randn(32, 65)  # batch of 32
        
        logits, mean, std = actor.forward(obs)
        
        assert logits.shape == (32, 5)
        assert mean.shape == (32, 4)
        assert std.shape == (32, 4)
    
    def test_actor_get_action(self):
        """Test action sampling."""
        actor = Actor(obs_dim=65)
        obs = torch.randn(1, 65)
        
        action, log_prob = actor.get_action(obs)
        
        assert "discrete" in action
        assert "continuous" in action
        assert action["discrete"].shape == (1,)
        assert action["continuous"].shape == (1, 4)
        assert log_prob.shape == (1,)
    
    def test_actor_deterministic(self):
        """Test deterministic action."""
        actor = Actor(obs_dim=65)
        obs = torch.randn(1, 65)
        
        action1, _ = actor.get_action(obs, deterministic=True)
        action2, _ = actor.get_action(obs, deterministic=True)
        
        # Deterministic actions should be the same
        assert action1["discrete"].item() == action2["discrete"].item()
        torch.testing.assert_close(action1["continuous"], action2["continuous"])


class TestCritic:
    """Tests for the Critic network."""
    
    def test_critic_creation(self):
        """Test critic initialization."""
        critic = Critic(state_dim=390)
        
        assert critic.state_dim == 390
    
    def test_critic_forward(self):
        """Test critic forward pass."""
        critic = Critic(state_dim=390)
        state = torch.randn(32, 390)
        
        value = critic.forward(state)
        
        assert value.shape == (32, 1)


class TestMAPPOPolicy:
    """Tests for the MAPPO Policy."""
    
    def test_policy_creation(self):
        """Test policy initialization."""
        policy = MAPPOPolicy(obs_dim=65, state_dim=390, device="cpu")
        
        assert policy.obs_dim == 65
        assert policy.state_dim == 390
    
    def test_policy_act(self):
        """Test policy action generation."""
        policy = MAPPOPolicy(obs_dim=65, state_dim=390, device="cpu")
        obs = np.random.randn(65).astype(np.float32)
        
        action, log_prob = policy.act(obs)
        
        assert "discrete" in action
        assert "continuous" in action
        assert isinstance(action["discrete"], np.ndarray)
        assert action["continuous"].shape == (4,)
    
    def test_policy_get_value(self):
        """Test value estimation."""
        policy = MAPPOPolicy(obs_dim=65, state_dim=390, device="cpu")
        state = np.random.randn(390).astype(np.float32)
        
        value = policy.get_value(state)
        
        assert isinstance(value, float)


class TestRolloutBuffer:
    """Tests for the Rollout Buffer."""
    
    def test_buffer_creation(self):
        """Test buffer initialization."""
        buffer = RolloutBuffer(
            buffer_size=100,
            obs_dim=65,
            state_dim=390,
            n_agents=6,
            device="cpu"
        )
        
        assert buffer.buffer_size == 100
        assert buffer.n_agents == 6
        assert buffer.ptr == 0
    
    def test_buffer_add(self):
        """Test adding transitions to buffer."""
        buffer = RolloutBuffer(
            buffer_size=100,
            obs_dim=65,
            state_dim=390,
            n_agents=6,
            device="cpu"
        )
        
        agent_ids = [f"agent_{i}" for i in range(6)]
        observations = {aid: np.random.randn(65).astype(np.float32) for aid in agent_ids}
        state = np.random.randn(390).astype(np.float32)
        actions = {aid: {"discrete": np.array(0), "continuous": np.zeros(4)} for aid in agent_ids}
        rewards = {aid: 0.0 for aid in agent_ids}
        dones = {aid: False for aid in agent_ids}
        log_probs = {aid: 0.0 for aid in agent_ids}
        values = {aid: 0.0 for aid in agent_ids}
        
        buffer.add(observations, state, actions, rewards, dones, log_probs, values, agent_ids)
        
        assert buffer.ptr == 1
    
    def test_buffer_is_full(self):
        """Test buffer full detection."""
        buffer = RolloutBuffer(
            buffer_size=5,
            obs_dim=65,
            state_dim=390,
            n_agents=2,
            device="cpu"
        )
        
        agent_ids = ["a0", "a1"]
        
        for _ in range(5):
            observations = {aid: np.random.randn(65).astype(np.float32) for aid in agent_ids}
            state = np.random.randn(390).astype(np.float32)
            actions = {aid: {"discrete": np.array(0), "continuous": np.zeros(4)} for aid in agent_ids}
            rewards = {aid: 0.0 for aid in agent_ids}
            dones = {aid: False for aid in agent_ids}
            log_probs = {aid: 0.0 for aid in agent_ids}
            values = {aid: 0.0 for aid in agent_ids}
            
            buffer.add(observations, state, actions, rewards, dones, log_probs, values, agent_ids)
        
        assert buffer.is_full()


class TestMAPPOAgent:
    """Tests for the MAPPO Agent."""
    
    def test_agent_creation(self):
        """Test agent initialization."""
        agent = MAPPOAgent(
            obs_dim=65,
            state_dim=390,
            n_agents=6,
            device="cpu"
        )
        
        assert agent.obs_dim == 65
        assert agent.n_agents == 6
    
    def test_agent_act(self):
        """Test agent action generation."""
        agent = MAPPOAgent(obs_dim=65, state_dim=390, n_agents=6, device="cpu")
        
        observations = {f"agent_{i}": np.random.randn(65).astype(np.float32) for i in range(6)}
        
        actions, log_probs = agent.act(observations)
        
        assert len(actions) == 6
        assert len(log_probs) == 6
        assert all("discrete" in a for a in actions.values())
        assert all("continuous" in a for a in actions.values())
    
    def test_agent_get_values(self):
        """Test value estimation."""
        agent = MAPPOAgent(obs_dim=65, state_dim=390, n_agents=6, device="cpu")
        
        state = np.random.randn(390).astype(np.float32)
        agent_ids = [f"agent_{i}" for i in range(6)]
        
        values = agent.get_values(state, agent_ids)
        
        assert len(values) == 6
        assert all(isinstance(v, float) for v in values.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
