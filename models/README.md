# Pretrained Models

This directory contains pretrained MAPPO models for SkyBattle.

## Available Models

| Model | Description | Win Rate |
|-------|-------------|----------|
| `mappo_easy.pt` | Easy difficulty AI | ~60% |
| `mappo_normal.pt` | Normal difficulty AI | ~75% |
| `mappo_hard.pt` | Hard difficulty AI | ~85% |

## Training Your Own

```bash
# Train a new model
python train.py --mode standard --episodes 1000

# The model will be saved to models/final_model.pt
```

## Loading Models

```python
from backend.agents import MAPPOAgent

agent = MAPPOAgent(obs_dim=65, state_dim=390, n_agents=6)
agent.load("models/mappo_normal.pt")
```
