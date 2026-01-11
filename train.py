#!/usr/bin/env python3
"""Training script for SkyBattle MAPPO agent."""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from backend.envs import CombatEnv, CombatConfig
from backend.agents import MAPPOAgent
from backend.utils.logger import TensorBoardLogger, ConsoleLogger


console = Console()


def train(args):
    """Main training loop."""
    console.print(f"[bold cyan]🎮 SkyBattle Training[/bold cyan]")
    console.print(f"Mode: {args.mode} | Episodes: {args.episodes} | Team Size: {args.team_size}")
    
    # Initialize loggers
    experiment_name = f"{args.mode}_{args.team_size}v{args.team_size}"
    try:
        tb_logger = TensorBoardLogger(log_dir=args.log_dir, experiment_name=experiment_name)
        console.print(f"[green]TensorBoard logging to: {args.log_dir}/{experiment_name}[/green]")
        console.print(f"[dim]Run: tensorboard --logdir {args.log_dir}[/dim]")
    except ImportError:
        tb_logger = None
        console.print("[yellow]TensorBoard not available, using console logging only[/yellow]")
    
    file_logger = ConsoleLogger(log_dir=args.output_dir, log_to_file=True)
    
    # Create environment
    config = CombatConfig(team_size=args.team_size)
    env = CombatEnv(config=config)
    
    obs_dim = env.obs_dim
    state_dim = obs_dim * args.team_size * 2
    n_agents = args.team_size * 2
    
    # Create agent
    agent = MAPPOAgent(
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_agents=n_agents,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        buffer_size=args.buffer_size,
    )
    
    # Training metrics
    episode_rewards = []
    win_counts = {"red": 0, "blue": 0, "draw": 0}
    
    # Training loop
    pbar = tqdm(range(args.episodes), desc="Training")
    
    for episode in pbar:
        observations, info = env.reset()
        agent_ids = list(observations.keys())
        state = np.concatenate([observations[aid] for aid in agent_ids])
        
        episode_reward = 0.0
        step = 0
        done = False
        
        while not done:
            # Get actions
            actions, log_probs = agent.act(observations)
            values = agent.get_values(state, agent_ids)
            
            # Step environment
            next_observations, rewards, terminated, truncated, info = env.step(actions)
            done = all(terminated.values()) or all(truncated.values())
            
            next_state = np.concatenate([next_observations[aid] for aid in agent_ids])
            
            # Store transition
            agent.store_transition(
                observations=observations,
                state=state,
                actions=actions,
                rewards=rewards,
                dones=terminated,
                log_probs=log_probs,
                values=values,
                agent_ids=agent_ids,
            )
            
            observations = next_observations
            state = next_state
            episode_reward += sum(rewards.values())
            step += 1
            
            # Update agent
            if agent.should_update():
                last_values = np.array([agent.policy.get_value(state) for _ in agent_ids])
                metrics = agent.update(last_values)
        
        # Record episode stats
        episode_rewards.append(episode_reward)
        winner = info.get("winner")
        if winner:
            win_counts[winner] += 1
        else:
            win_counts["draw"] += 1
        
        # Calculate metrics
        avg_reward = np.mean(episode_rewards[-100:])
        win_rate = win_counts["red"] / (episode + 1) * 100
        
        # Log to TensorBoard
        if tb_logger:
            tb_logger.log_episode(episode + 1, {
                "reward": episode_reward,
                "reward_avg_100": avg_reward,
                "steps": step,
                "win_rate_red": win_rate,
                "red_alive": info.get("red_alive", 0),
                "blue_alive": info.get("blue_alive", 0),
            })
        
        # Log to file
        file_logger.log_episode(episode + 1, {
            "reward": episode_reward,
            "steps": step,
            "winner": winner or "draw",
        })
        
        # Update progress bar
        pbar.set_postfix({
            "reward": f"{avg_reward:.1f}",
            "win_rate": f"{win_rate:.1f}%",
            "steps": step,
        })
        
        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            save_path = Path(args.output_dir) / f"checkpoint_{episode + 1}.pt"
            agent.save(str(save_path))
            console.print(f"[green]Saved checkpoint: {save_path}[/green]")
    
    # Final save
    final_path = Path(args.output_dir) / "final_model.pt"
    agent.save(str(final_path))
    
    # Save metrics
    file_logger.save_metrics()
    
    # Close TensorBoard logger
    if tb_logger:
        # Log hyperparameters
        tb_logger.log_hyperparameters({
            "algorithm": "MAPPO",
            "team_size": args.team_size,
            "lr_actor": args.lr_actor,
            "lr_critic": args.lr_critic,
            "episodes": args.episodes,
            "buffer_size": args.buffer_size,
        }, {
            "final_reward": np.mean(episode_rewards[-100:]),
            "final_win_rate": win_counts["red"] / args.episodes,
        })
        tb_logger.close()
    
    # Print summary
    console.print("\n[bold green]✅ Training Complete![/bold green]")
    
    table = Table(title="Training Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Episodes", str(args.episodes))
    table.add_row("Mean Reward (last 100)", f"{np.mean(episode_rewards[-100:]):.2f}")
    table.add_row("Red Wins", str(win_counts["red"]))
    table.add_row("Blue Wins", str(win_counts["blue"]))
    table.add_row("Draws", str(win_counts["draw"]))
    table.add_row("Model Saved", str(final_path))
    
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Train SkyBattle MAPPO agent")
    
    parser.add_argument("--mode", type=str, default="standard", choices=["quick", "standard", "full"],
                        help="Training mode")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes")
    parser.add_argument("--team-size", type=int, default=3, help="Number of drones per team")
    parser.add_argument("--lr-actor", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--lr-critic", type=float, default=5e-4, help="Critic learning rate")
    parser.add_argument("--buffer-size", type=int, default=2048, help="Rollout buffer size")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--save-interval", type=int, default=100, help="Checkpoint interval")
    parser.add_argument("--self-play", action="store_true", help="Enable self-play training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="runs", help="TensorBoard log directory")
    
    args = parser.parse_args()
    
    # Mode presets
    if args.episodes is None:
        mode_episodes = {"quick": 100, "standard": 1000, "full": 5000}
        args.episodes = mode_episodes[args.mode]
    
    # Set seed
    np.random.seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train(args)


if __name__ == "__main__":
    main()
