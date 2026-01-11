"""Weapon systems for drones."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Projectile:
    """Base projectile class."""
    id: str
    owner_id: str
    owner_team: str
    position: np.ndarray
    velocity: np.ndarray
    damage: float
    lifetime: float
    
    def update(self, dt: float) -> bool:
        self.position += self.velocity * dt
        self.lifetime -= dt
        return self.lifetime > 0


@dataclass
class Bullet(Projectile):
    """Machine gun bullet."""
    pass


@dataclass
class MissileProjectile(Projectile):
    """Homing missile."""
    target_id: Optional[str] = None
    tracking: float = 0.8
    
    def update_tracking(self, target_pos: Optional[np.ndarray], dt: float):
        if target_pos is None or self.target_id is None:
            return
        
        to_target = target_pos - self.position
        dist = np.linalg.norm(to_target)
        if dist < 1e-6:
            return
        
        desired = to_target / dist
        speed = np.linalg.norm(self.velocity)
        if speed < 1e-6:
            return
        
        current = self.velocity / speed
        new_dir = current * (1 - self.tracking * dt) + desired * self.tracking * dt
        new_dir = new_dir / np.linalg.norm(new_dir)
        self.velocity = new_dir * speed


@dataclass
class Flare:
    """Countermeasure flare."""
    id: str
    position: np.ndarray
    owner_id: str
    lifetime: float = 3.0
    radius: float = 50.0
    
    def update(self, dt: float) -> bool:
        self.lifetime -= dt
        self.position[2] -= 5.0 * dt
        return self.lifetime > 0
    
    def can_distract(self, missile: MissileProjectile) -> bool:
        return np.linalg.norm(missile.position - self.position) < self.radius
