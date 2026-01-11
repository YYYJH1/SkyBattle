"""Tests for the Combat Environment."""

import pytest
import numpy as np
from backend.envs import CombatEnv, CombatConfig, Drone, DroneAction


class TestDrone:
    """Tests for the Drone class."""
    
    def test_drone_creation(self):
        """Test drone initialization."""
        drone = Drone("test_0", "red")
        
        assert drone.id == "test_0"
        assert drone.team == "red"
        assert drone.hp == 100.0
        assert drone.shield == 50.0
        assert drone.energy == 100.0
        assert drone.is_alive is True
    
    def test_drone_with_position(self):
        """Test drone with custom position."""
        pos = np.array([100.0, 50.0, 80.0])
        ori = np.array([0.0, 0.1, 0.5])
        drone = Drone("test_1", "blue", pos, ori)
        
        np.testing.assert_array_equal(drone.position, pos)
        np.testing.assert_array_equal(drone.orientation, ori)
    
    def test_drone_take_damage(self):
        """Test damage handling."""
        drone = Drone("test_2", "red")
        
        # Damage absorbed by shield first
        killed = drone.take_damage(30.0)
        assert killed is False
        assert drone.shield == 20.0
        assert drone.hp == 100.0
        
        # More damage goes to HP
        killed = drone.take_damage(50.0)
        assert killed is False
        assert drone.shield == 0.0
        assert drone.hp == 70.0
        
        # Kill the drone
        killed = drone.take_damage(100.0)
        assert killed is True
        assert drone.is_alive is False
        assert drone.hp == 0.0
    
    def test_drone_apply_action(self):
        """Test action application."""
        drone = Drone("test_3", "red")
        action = DroneAction(
            discrete=0,  # idle
            continuous=np.array([1.0, 0.0, 0.0, 0.0])  # full throttle
        )
        
        initial_pos = drone.position.copy()
        events = drone.apply_action(action, dt=0.1)
        
        # Drone should have moved forward
        assert not np.array_equal(drone.position, initial_pos)
        assert drone.velocity[0] > 0  # Moving forward
    
    def test_drone_fire_gun(self):
        """Test gun firing."""
        drone = Drone("test_4", "red")
        initial_ammo = drone.ammo
        
        action = DroneAction(discrete=1, continuous=np.zeros(4))
        events = drone.apply_action(action, dt=0.1)
        
        assert "fire_gun" in events
        assert drone.ammo == initial_ammo - 1
    
    def test_drone_distance(self):
        """Test distance calculation."""
        drone1 = Drone("d1", "red", np.array([0.0, 0.0, 0.0]))
        drone2 = Drone("d2", "blue", np.array([100.0, 0.0, 0.0]))
        
        distance = drone1.distance_to(drone2)
        assert distance == 100.0


class TestCombatEnv:
    """Tests for the Combat Environment."""
    
    def test_env_creation(self):
        """Test environment creation."""
        config = CombatConfig(team_size=3)
        env = CombatEnv(config=config)
        
        assert env.n_agents == 6
        assert env.config.team_size == 3
    
    def test_env_reset(self):
        """Test environment reset."""
        env = CombatEnv(config=CombatConfig(team_size=2))
        observations, info = env.reset(seed=42)
        
        assert len(observations) == 4  # 2v2
        assert "red_0" in observations
        assert "blue_0" in observations
        assert info["red_alive"] == 2
        assert info["blue_alive"] == 2
    
    def test_env_step(self):
        """Test environment step."""
        env = CombatEnv(config=CombatConfig(team_size=2))
        observations, _ = env.reset(seed=42)
        
        # Create actions for all agents
        actions = {}
        for agent_id in observations:
            actions[agent_id] = {
                "discrete": 0,  # idle
                "continuous": [0.5, 0.0, 0.0, 0.0],  # some throttle
            }
        
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        
        assert len(next_obs) == 4
        assert len(rewards) == 4
        assert info["step"] == 1
    
    def test_env_observation_shape(self):
        """Test observation dimensions."""
        config = CombatConfig(team_size=3)
        env = CombatEnv(config=config)
        observations, _ = env.reset()
        
        for obs in observations.values():
            assert obs.shape == (env.obs_dim,)
            assert obs.dtype == np.float32
    
    def test_env_gymnasium_compatible(self):
        """Test Gymnasium compatibility."""
        env = CombatEnv()
        
        # Check spaces are defined
        assert env.action_space is not None
        assert env.observation_space is not None
        
        # Check reset returns correct format
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)


class TestCombatConfig:
    """Tests for CombatConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = CombatConfig()
        
        assert config.team_size == 3
        assert config.dt == 0.1
        assert config.max_steps == 3000
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CombatConfig(
            team_size=5,
            max_steps=1000,
            damage_reward=1.0,
        )
        
        assert config.team_size == 5
        assert config.max_steps == 1000
        assert config.damage_reward == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
