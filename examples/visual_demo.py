#!/usr/bin/env python3
"""
SkyBattle Visual Demo - Matplotlib-based visualization
Shows drone combat in real-time using 3D plotting
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

from backend.envs import CombatEnv, CombatConfig
from backend.agents import MAPPOAgent


def run_visual_demo():
    """Run a visual demonstration of drone combat."""
    
    print("🎮 SkyBattle Visual Demo")
    print("=" * 50)
    
    # Create environment
    config = CombatConfig(team_size=3, max_steps=500)
    env = CombatEnv(config=config)
    
    # Create agent
    agent = MAPPOAgent(
        obs_dim=env.obs_dim,
        state_dim=env.obs_dim * 6,
        n_agents=6,
        device='cpu'
    )
    
    # Setup figure
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10))
    
    # Main 3D view
    ax3d = fig.add_subplot(221, projection='3d')
    ax3d.set_xlim(-500, 500)
    ax3d.set_ylim(-500, 500)
    ax3d.set_zlim(0, 300)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z (Height)')
    ax3d.set_title('🎮 SkyBattle Arena', fontsize=14, fontweight='bold')
    
    # Top-down view
    ax_top = fig.add_subplot(222)
    ax_top.set_xlim(-500, 500)
    ax_top.set_ylim(-500, 500)
    ax_top.set_xlabel('X')
    ax_top.set_ylabel('Y')
    ax_top.set_title('📍 Top-Down View', fontsize=12)
    ax_top.set_aspect('equal')
    ax_top.grid(True, alpha=0.3)
    
    # Stats panel
    ax_stats = fig.add_subplot(223)
    ax_stats.axis('off')
    ax_stats.set_title('📊 Battle Stats', fontsize=12)
    
    # HP bars
    ax_hp = fig.add_subplot(224)
    ax_hp.set_title('❤️ Health Status', fontsize=12)
    
    plt.tight_layout()
    
    # Initialize
    obs, info = env.reset(seed=42)
    
    # Storage for animation
    history = {'red_hp': [], 'blue_hp': [], 'steps': []}
    
    def update(frame):
        nonlocal obs
        
        # Clear axes
        ax3d.cla()
        ax_top.cla()
        ax_stats.cla()
        ax_hp.cla()
        
        # Get actions and step
        actions, _ = agent.act(obs)
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Get state for rendering
        state = env.get_state_for_render()
        
        # Collect HP data
        red_hp = sum(d['hp'] for d in state['drones'] if d['team'] == 'red' and d['is_alive'])
        blue_hp = sum(d['hp'] for d in state['drones'] if d['team'] == 'blue' and d['is_alive'])
        history['red_hp'].append(red_hp)
        history['blue_hp'].append(blue_hp)
        history['steps'].append(frame)
        
        # === 3D View ===
        ax3d.set_xlim(-500, 500)
        ax3d.set_ylim(-500, 500)
        ax3d.set_zlim(0, 300)
        ax3d.set_xlabel('X', fontsize=10)
        ax3d.set_ylabel('Y', fontsize=10)
        ax3d.set_zlabel('Height', fontsize=10)
        
        for drone in state['drones']:
            pos = drone['position']
            color = '#FF4444' if drone['team'] == 'red' else '#4444FF'
            marker = 'o' if drone['is_alive'] else 'x'
            size = 100 if drone['is_alive'] else 50
            alpha = 1.0 if drone['is_alive'] else 0.3
            
            ax3d.scatter(pos[0], pos[1], pos[2], c=color, s=size, 
                        marker=marker, alpha=alpha, edgecolors='white', linewidth=1)
            
            # Draw velocity vector
            if drone['is_alive']:
                vel = drone['velocity']
                scale = 0.5
                ax3d.quiver(pos[0], pos[1], pos[2], 
                           vel[0]*scale, vel[1]*scale, vel[2]*scale,
                           color=color, alpha=0.6, arrow_length_ratio=0.3)
        
        # Draw projectiles
        for proj in state['projectiles']:
            pos = proj['position']
            ax3d.scatter(pos[0], pos[1], pos[2], c='yellow', s=20, marker='.')
        
        ax3d.set_title(f'🎮 SkyBattle Arena - Step {frame}', fontsize=14, fontweight='bold')
        
        # === Top-Down View ===
        ax_top.set_xlim(-500, 500)
        ax_top.set_ylim(-500, 500)
        ax_top.set_xlabel('X')
        ax_top.set_ylabel('Y')
        ax_top.grid(True, alpha=0.3)
        ax_top.set_aspect('equal')
        
        for drone in state['drones']:
            pos = drone['position']
            color = '#FF4444' if drone['team'] == 'red' else '#4444FF'
            if drone['is_alive']:
                circle = plt.Circle((pos[0], pos[1]), 20, color=color, alpha=0.7)
                ax_top.add_patch(circle)
                # Show HP text
                ax_top.text(pos[0], pos[1]+30, f"{drone['hp']:.0f}", 
                           ha='center', va='bottom', fontsize=8, color='white')
            else:
                ax_top.scatter(pos[0], pos[1], c='gray', s=50, marker='x', alpha=0.5)
        
        ax_top.set_title('📍 Top-Down View', fontsize=12)
        
        # Legend
        red_patch = mpatches.Patch(color='#FF4444', label='Red Team')
        blue_patch = mpatches.Patch(color='#4444FF', label='Blue Team')
        ax_top.legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=8)
        
        # === Stats Panel ===
        ax_stats.axis('off')
        
        red_alive = info['red_alive']
        blue_alive = info['blue_alive']
        
        stats_text = f"""
        ╔══════════════════════════════════╗
        ║        BATTLE STATISTICS         ║
        ╠══════════════════════════════════╣
        ║  Step: {frame:>4} / {config.max_steps:<4}              ║
        ╠══════════════════════════════════╣
        ║  🔴 RED TEAM                     ║
        ║     Alive: {red_alive} / {config.team_size}                   ║
        ║     Total HP: {red_hp:.0f}                 ║
        ╠══════════════════════════════════╣
        ║  🔵 BLUE TEAM                    ║
        ║     Alive: {blue_alive} / {config.team_size}                   ║
        ║     Total HP: {blue_hp:.0f}                 ║
        ╚══════════════════════════════════╝
        """
        
        ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, fontfamily='monospace', verticalalignment='center',
                     color='#00FF00')
        ax_stats.set_title('📊 Battle Stats', fontsize=12)
        
        # === HP History ===
        if len(history['steps']) > 1:
            ax_hp.plot(history['steps'], history['red_hp'], 'r-', linewidth=2, label='Red Team')
            ax_hp.plot(history['steps'], history['blue_hp'], 'b-', linewidth=2, label='Blue Team')
            ax_hp.fill_between(history['steps'], history['red_hp'], alpha=0.2, color='red')
            ax_hp.fill_between(history['steps'], history['blue_hp'], alpha=0.2, color='blue')
        
        ax_hp.set_xlabel('Step')
        ax_hp.set_ylabel('Total HP')
        ax_hp.set_title('❤️ Health Over Time', fontsize=12)
        ax_hp.legend(loc='upper right', fontsize=8)
        ax_hp.grid(True, alpha=0.3)
        ax_hp.set_xlim(0, max(frame, 10))
        ax_hp.set_ylim(0, config.team_size * 100 + 50)
        
        # Check game end
        if all(terminated.values()):
            winner = info.get('winner', 'Draw')
            ax3d.text2D(0.5, 0.5, f"GAME OVER!\n{winner.upper()} WINS!", 
                       transform=ax3d.transAxes, fontsize=20, fontweight='bold',
                       ha='center', va='center', color='gold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        return []
    
    print("🚀 Starting visualization...")
    print("   Close the window to stop")
    print()
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=config.max_steps, 
                       interval=100, blit=False, repeat=False)
    
    plt.show()
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    run_visual_demo()
