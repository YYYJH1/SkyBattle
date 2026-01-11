"""Drone model with physics and state management."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DroneState:
    """Immutable snapshot of drone state."""
    id: str
    team: str
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    hp: float
    shield: float
    energy: float
    ammo: int
    missiles: int
    is_alive: bool
    
    def to_array(self) -> np.ndarray:
        return np.concatenate([
            self.position, self.velocity, self.orientation,
            [self.hp / 100.0, self.shield / 50.0, self.energy / 100.0, self.ammo / 500.0],
        ]).astype(np.float32)


@dataclass
class DroneAction:
    """Action taken by a drone."""
    discrete: int  # 0:idle, 1:fire_gun, 2:fire_missile, 3:flare, 4:boost
    continuous: np.ndarray  # [throttle, pitch, yaw, roll]
    
    @classmethod
    def idle(cls) -> "DroneAction":
        return cls(discrete=0, continuous=np.zeros(4, dtype=np.float32))


class Drone:
    """Drone entity with physics simulation."""
    
    MAX_SPEED = 200.0
    MAX_ACCELERATION = 50.0
    MAX_TURN_RATE = 2.0
    DRAG = 0.02
    
    MAX_HP = 100.0
    MAX_SHIELD = 50.0
    MAX_ENERGY = 100.0
    MAX_AMMO = 500
    MAX_MISSILES = 4
    
    SHIELD_REGEN = 2.0
    ENERGY_REGEN = 5.0
    BOOST_COST = 20.0
    MISSILE_COST = 15.0
    FLARE_COST = 10.0
    
    def __init__(self, drone_id: str, team: str,
                 position: Optional[np.ndarray] = None,
                 orientation: Optional[np.ndarray] = None):
        self.id = drone_id
        self.team = team
        self.position = position if position is not None else np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = orientation if orientation is not None else np.zeros(3, dtype=np.float32)
        
        self.hp = self.MAX_HP
        self.shield = self.MAX_SHIELD
        self.energy = self.MAX_ENERGY
        self.ammo = self.MAX_AMMO
        self.missiles = self.MAX_MISSILES
        self.is_alive = True
        self.is_boosting = False
        
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.kills = 0
        
        self.missile_cooldown = 0.0
        self.flare_cooldown = 0.0
    
    def reset(self, position: np.ndarray, orientation: np.ndarray):
        self.position = position.copy()
        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = orientation.copy()
        self.hp = self.MAX_HP
        self.shield = self.MAX_SHIELD
        self.energy = self.MAX_ENERGY
        self.ammo = self.MAX_AMMO
        self.missiles = self.MAX_MISSILES
        self.is_alive = True
        self.is_boosting = False
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.kills = 0
        self.missile_cooldown = 0.0
        self.flare_cooldown = 0.0
    
    def get_state(self) -> DroneState:
        return DroneState(
            id=self.id, team=self.team,
            position=self.position.copy(), velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            hp=self.hp, shield=self.shield, energy=self.energy,
            ammo=self.ammo, missiles=self.missiles, is_alive=self.is_alive,
        )
    
    def apply_action(self, action: DroneAction, dt: float) -> dict:
        if not self.is_alive:
            return {}
        
        events = {}
        self.missile_cooldown = max(0, self.missile_cooldown - dt)
        self.flare_cooldown = max(0, self.flare_cooldown - dt)
        
        # Discrete actions
        if action.discrete == 1 and self.ammo > 0:
            self.ammo -= 1
            events["fire_gun"] = True
        elif action.discrete == 2 and self.missiles > 0 and self.missile_cooldown <= 0 and self.energy >= self.MISSILE_COST:
            self.missiles -= 1
            self.energy -= self.MISSILE_COST
            self.missile_cooldown = 5.0
            events["fire_missile"] = True
        elif action.discrete == 3 and self.flare_cooldown <= 0 and self.energy >= self.FLARE_COST:
            self.energy -= self.FLARE_COST
            self.flare_cooldown = 8.0
            events["deploy_flare"] = True
        elif action.discrete == 4 and self.energy >= self.BOOST_COST * dt:
            self.energy -= self.BOOST_COST * dt
            self.is_boosting = True
        else:
            self.is_boosting = False
        
        # Continuous control
        throttle = np.clip(action.continuous[0], -1, 1)
        pitch_rate = np.clip(action.continuous[1], -1, 1) * self.MAX_TURN_RATE
        yaw_rate = np.clip(action.continuous[2], -1, 1) * self.MAX_TURN_RATE
        roll_rate = np.clip(action.continuous[3], -1, 1) * self.MAX_TURN_RATE
        
        self.orientation[0] += roll_rate * dt
        self.orientation[1] += pitch_rate * dt
        self.orientation[2] += yaw_rate * dt
        self.orientation = np.mod(self.orientation + np.pi, 2 * np.pi) - np.pi
        
        # Forward direction
        yaw, pitch = self.orientation[2], self.orientation[1]
        forward = np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ], dtype=np.float32)
        
        # Acceleration
        accel_mag = throttle * self.MAX_ACCELERATION * (1.5 if self.is_boosting else 1.0)
        acceleration = forward * accel_mag
        
        # Drag
        speed = np.linalg.norm(self.velocity)
        if speed > 0:
            acceleration -= self.DRAG * speed * self.velocity
        
        self.velocity += acceleration * dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.MAX_SPEED:
            self.velocity = self.velocity / speed * self.MAX_SPEED
        
        self.position += self.velocity * dt
        
        # Regeneration
        if self.shield < self.MAX_SHIELD:
            self.shield = min(self.MAX_SHIELD, self.shield + self.SHIELD_REGEN * dt)
        if self.energy < self.MAX_ENERGY:
            regen = self.ENERGY_REGEN * (1.5 if speed < 60 else 1.0)
            self.energy = min(self.MAX_ENERGY, self.energy + regen * dt)
        
        return events
    
    def take_damage(self, damage: float) -> bool:
        if not self.is_alive:
            return False
        
        self.damage_taken += damage
        if self.shield > 0:
            absorbed = min(self.shield, damage)
            self.shield -= absorbed
            damage -= absorbed
        
        if damage > 0:
            self.hp -= damage
        
        if self.hp <= 0:
            self.hp = 0
            self.is_alive = False
            return True
        return False
    
    def get_forward(self) -> np.ndarray:
        yaw, pitch = self.orientation[2], self.orientation[1]
        return np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], dtype=np.float32)
    
    def distance_to(self, other: "Drone") -> float:
        return float(np.linalg.norm(self.position - other.position))
    
    def angle_to(self, other: "Drone") -> float:
        to_other = other.position - self.position
        dist = np.linalg.norm(to_other)
        if dist < 1e-6:
            return 0.0
        dot = np.clip(np.dot(self.get_forward(), to_other / dist), -1, 1)
        return float(np.arccos(dot))
    
    def can_see(self, other: "Drone", fov: float = np.pi / 3) -> bool:
        return self.angle_to(other) <= fov / 2
