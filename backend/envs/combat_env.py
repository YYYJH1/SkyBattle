"""Multi-Agent Drone Combat Environment."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .drone import Drone, DroneAction
from .weapons import Bullet, MissileProjectile, Flare


@dataclass
class CombatConfig:
    """Combat environment configuration."""
    team_size: int = 3
    map_bounds: Tuple[float, float] = (-500.0, 500.0)
    map_height: Tuple[float, float] = (0.0, 300.0)
    dt: float = 0.1
    max_steps: int = 3000
    
    # Reward weights
    damage_reward: float = 0.5
    kill_reward: float = 50.0
    damage_penalty: float = 0.3
    death_penalty: float = 30.0
    survival_reward: float = 0.1
    team_weight: float = 0.3
    
    bullet_hit_radius: float = 12.0  # 增大命中范围
    missile_hit_radius: float = 15.0


class CombatEnv(gym.Env):
    """Multi-Agent Drone Combat Environment."""
    
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(self, config: Optional[CombatConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config or CombatConfig()
        self.render_mode = render_mode
        
        # Observation dimensions
        self.obs_dim = 13 + self.config.team_size * 10 + (self.config.team_size - 1) * 8 + 6
        self.n_agents = self.config.team_size * 2
        
        # Action/observation spaces
        self.action_space = spaces.Dict({
            "discrete": spaces.Discrete(5),
            "continuous": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        })
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        # State
        self.drones: Dict[str, Drone] = {}
        self.projectiles: List = []
        self.flares: List[Flare] = []
        self.step_count = 0
        self.wind = np.zeros(3, dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.projectiles = []
        self.flares = []
        self.drones = {}
        
        spawn_offset, height = 120.0, 100.0  # 缩短初始距离，让战斗更快开始
        
        # Spawn red team
        for i in range(self.config.team_size):
            drone_id = f"red_{i}"
            pos = np.array([-spawn_offset, (i - self.config.team_size / 2) * 50, height], dtype=np.float32)
            ori = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.drones[drone_id] = Drone(drone_id, "red", pos, ori)
        
        # Spawn blue team
        for i in range(self.config.team_size):
            drone_id = f"blue_{i}"
            pos = np.array([spawn_offset, (i - self.config.team_size / 2) * 50, height], dtype=np.float32)
            ori = np.array([0.0, 0.0, np.pi], dtype=np.float32)
            self.drones[drone_id] = Drone(drone_id, "blue", pos, ori)
        
        if self.np_random:
            self.wind = self.np_random.uniform(-5, 5, 3).astype(np.float32)
        
        return self._get_observations(), self._get_info()
    
    def step(self, actions: Dict[str, Dict[str, Any]]):
        self.step_count += 1
        dt = self.config.dt
        events = []
        
        # Process actions
        for drone_id, action_dict in actions.items():
            if drone_id not in self.drones:
                continue
            drone = self.drones[drone_id]
            if not drone.is_alive:
                continue
            
            action = DroneAction(
                discrete=action_dict.get("discrete", 0),
                continuous=np.array(action_dict.get("continuous", [0, 0, 0, 0]), dtype=np.float32),
            )
            drone_events = drone.apply_action(action, dt)
            
            if drone_events.get("fire_gun"):
                self._fire_gun(drone)
            if drone_events.get("fire_missile"):
                self._fire_missile(drone)
            if drone_events.get("deploy_flare"):
                self.flares.append(Flare(f"flare_{len(self.flares)}", drone.position.copy(), drone.id))
        
        # Update projectiles
        self._update_projectiles(dt)
        
        # Check collisions
        hit_events = self._check_collisions()
        events.extend(hit_events)
        
        # Update flares
        self.flares = [f for f in self.flares if f.update(dt)]
        
        # Check boundaries
        self._check_boundaries()
        
        # Compute rewards
        rewards = self._compute_rewards(events)
        
        # Check done
        terminated, truncated = self._check_done()
        
        return self._get_observations(), rewards, terminated, truncated, self._get_info()
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        return {drone_id: self._get_obs(drone) for drone_id, drone in self.drones.items()}
    
    def _get_obs(self, drone: Drone) -> np.ndarray:
        parts = []
        
        # Self state (13)
        parts.append(np.concatenate([
            drone.position / 500.0, drone.velocity / 200.0, drone.orientation / np.pi,
            [drone.hp / 100.0, drone.shield / 50.0, drone.energy / 100.0, drone.ammo / 500.0],
        ]))
        
        # Enemies (team_size * 10)
        enemies = sorted([d for d in self.drones.values() if d.team != drone.team and d.is_alive],
                        key=lambda e: drone.distance_to(e))[:self.config.team_size]
        for i in range(self.config.team_size):
            if i < len(enemies):
                e = enemies[i]
                parts.append(np.array([
                    *(e.position - drone.position) / 500.0, *(e.velocity - drone.velocity) / 200.0,
                    drone.distance_to(e) / 1000.0, drone.angle_to(e) / np.pi, e.hp / 100.0, 0.0,
                ], dtype=np.float32))
            else:
                parts.append(np.zeros(10, dtype=np.float32))
        
        # Allies ((team_size-1) * 8)
        allies = sorted([d for d in self.drones.values() if d.team == drone.team and d.id != drone.id and d.is_alive],
                       key=lambda a: drone.distance_to(a))[:self.config.team_size - 1]
        for i in range(self.config.team_size - 1):
            if i < len(allies):
                a = allies[i]
                parts.append(np.array([
                    *(a.position - drone.position) / 500.0, *(a.velocity - drone.velocity) / 200.0,
                    a.hp / 100.0, 1.0,
                ], dtype=np.float32))
            else:
                parts.append(np.zeros(8, dtype=np.float32))
        
        # Environment (6)
        bounds = self.config.map_bounds
        parts.append(np.concatenate([
            self.wind / 10.0,
            [(drone.position[i] - bounds[0]) / (bounds[1] - bounds[0]) for i in range(3)],
        ]))
        
        return np.concatenate(parts).astype(np.float32)
    
    def _fire_gun(self, drone: Drone):
        direction = drone.get_forward() + np.random.uniform(-0.08, 0.08, 3)  # 更大散布
        direction = direction / np.linalg.norm(direction)
        self.projectiles.append(Bullet(
            id=f"bullet_{len(self.projectiles)}", owner_id=drone.id, owner_team=drone.team,
            position=drone.position.copy() + direction * 5, velocity=direction * 600.0 + drone.velocity,
            damage=8.0, lifetime=1.2,  # 更快、更久、更痛
        ))
    
    def _fire_missile(self, drone: Drone):
        enemies = [d for d in self.drones.values() if d.team != drone.team and d.is_alive]
        target = min(enemies, key=lambda e: drone.distance_to(e)) if enemies else None
        direction = drone.get_forward()
        self.projectiles.append(MissileProjectile(
            id=f"missile_{len(self.projectiles)}", owner_id=drone.id, owner_team=drone.team,
            position=drone.position.copy() + direction * 5, velocity=direction * 150.0,
            damage=40.0, lifetime=3.5, target_id=target.id if target else None,
        ))
    
    def _update_projectiles(self, dt: float):
        active = []
        for proj in self.projectiles:
            if isinstance(proj, MissileProjectile) and proj.target_id:
                target = self.drones.get(proj.target_id)
                distracted = any(f.can_distract(proj) for f in self.flares)
                if distracted:
                    proj.target_id = None
                elif target and target.is_alive:
                    proj.update_tracking(target.position, dt)
            if proj.update(dt):
                active.append(proj)
        self.projectiles = active
    
    def _check_collisions(self) -> List[dict]:
        events = []
        remaining = []
        
        for proj in self.projectiles:
            hit = False
            hit_radius = self.config.missile_hit_radius if isinstance(proj, MissileProjectile) else self.config.bullet_hit_radius
            
            for drone in self.drones.values():
                if not drone.is_alive or drone.team == proj.owner_team:
                    continue
                if np.linalg.norm(proj.position - drone.position) < hit_radius:
                    killed = drone.take_damage(proj.damage)
                    attacker = self.drones.get(proj.owner_id)
                    if attacker:
                        attacker.damage_dealt += proj.damage
                        if killed:
                            attacker.kills += 1
                    events.append({"type": "kill" if killed else "hit", "attacker": proj.owner_id,
                                  "target": drone.id, "damage": proj.damage})
                    hit = True
                    break
            if not hit:
                remaining.append(proj)
        
        self.projectiles = remaining
        return events
    
    def _check_boundaries(self):
        bounds = self.config.map_bounds
        height = self.config.map_height
        
        for drone in self.drones.values():
            if not drone.is_alive:
                continue
            for i in range(2):
                if drone.position[i] < bounds[0]:
                    drone.position[i] = bounds[0]
                    drone.velocity[i] = abs(drone.velocity[i]) * 0.5
                elif drone.position[i] > bounds[1]:
                    drone.position[i] = bounds[1]
                    drone.velocity[i] = -abs(drone.velocity[i]) * 0.5
            if drone.position[2] < height[0]:
                drone.position[2] = height[0]
                drone.velocity[2] = abs(drone.velocity[2]) * 0.5
                drone.take_damage(5.0)
            elif drone.position[2] > height[1]:
                drone.position[2] = height[1]
                drone.velocity[2] = -abs(drone.velocity[2]) * 0.5
    
    def _compute_rewards(self, events: List[dict]) -> Dict[str, float]:
        cfg = self.config
        rewards = {d: cfg.survival_reward if self.drones[d].is_alive else 0.0 for d in self.drones}
        
        for event in events:
            attacker, target = event.get("attacker"), event.get("target")
            damage = event.get("damage", 0)
            if attacker in rewards:
                rewards[attacker] += damage * cfg.damage_reward
                if event["type"] == "kill":
                    rewards[attacker] += cfg.kill_reward
            if target in rewards:
                rewards[target] -= damage * cfg.damage_penalty
                if event["type"] == "kill":
                    rewards[target] -= cfg.death_penalty
        
        return rewards
    
    def _check_done(self):
        red_alive = any(d.is_alive for d in self.drones.values() if d.team == "red")
        blue_alive = any(d.is_alive for d in self.drones.values() if d.team == "blue")
        
        terminated = {d: not red_alive or not blue_alive for d in self.drones}
        truncated = {d: self.step_count >= self.config.max_steps for d in self.drones}
        
        return terminated, truncated
    
    def _get_info(self) -> dict:
        red_alive = sum(1 for d in self.drones.values() if d.team == "red" and d.is_alive)
        blue_alive = sum(1 for d in self.drones.values() if d.team == "blue" and d.is_alive)
        return {
            "step": self.step_count, "red_alive": red_alive, "blue_alive": blue_alive,
            "winner": "red" if blue_alive == 0 and red_alive > 0 else "blue" if red_alive == 0 and blue_alive > 0 else None,
        }
    
    def get_state_for_render(self) -> dict:
        return {
            "step": self.step_count,
            "drones": [{"id": d.id, "team": d.team, "position": d.position.tolist(),
                       "velocity": d.velocity.tolist(), "orientation": d.orientation.tolist(),
                       "hp": d.hp, "shield": d.shield, "is_alive": d.is_alive} for d in self.drones.values()],
            "projectiles": [{"id": p.id, "position": p.position.tolist()} for p in self.projectiles],
        }
    
    def render(self):
        if self.render_mode == "human":
            info = self._get_info()
            print(f"Step {self.step_count}: Red({info['red_alive']}) vs Blue({info['blue_alive']})")
