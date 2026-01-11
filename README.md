# ✈️ SkyBattle - Multi-Agent Drone Combat Simulator

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Vue 3](https://img.shields.io/badge/vue-3.x-4FC08D?style=for-the-badge&logo=vue.js&logoColor=white)](https://vuejs.org/)
[![Three.js](https://img.shields.io/badge/three.js-r160-000000?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

**🎮 Watch AI drones battle it out, or challenge them yourself!**

[🎮 Live Demo](#-demo) · [🚀 Quick Start](#-quick-start) · [📖 Documentation](docs/) · [🤝 Contributing](#-contributing)

</div>

---

## 🌟 What is SkyBattle?

**SkyBattle** is an interactive drone combat simulation platform powered by **Multi-Agent Reinforcement Learning (MARL)**. It transforms cutting-edge AI research into an engaging, visual experience where you can:

- 👁️ **Watch** AI-controlled drone squadrons battle each other
- 🎮 **Play** against trained AI in human vs machine combat
- 🔬 **Train** your own AI agents with customizable parameters
- 🏆 **Compete** on the leaderboard with your trained models

---

## ✨ Features

| Mode | Description |
|------|-------------|
| 👁️ **AI vs AI** | Watch two AI teams battle - perfect for learning and entertainment |
| 🎮 **Human vs AI** | Take control of a drone and challenge the AI |
| 🔬 **Training** | Train your own MAPPO agent with real-time visualization |
| 🏆 **Tournament** | Upload your model and compete on the global leaderboard |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) CUDA 12.x for GPU training

### Installation

```bash
# Clone the repository
git clone https://github.com/YYYJH1/SkyBattle.git
cd SkyBattle

# Create Python environment
conda create -n skybattle python=3.10
conda activate skybattle

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Start Playing

```bash
# Terminal 1: Start the backend server
python server.py --port 8000

# Terminal 2: Start the frontend
cd frontend && npm run dev
```

Then open http://localhost:5173 🎮

---

## 🤖 Training Your Own Agent

```bash
# Quick training (10 min)
python train.py --mode quick --episodes 100

# Standard training (1-2 hours)
python train.py --mode standard --episodes 1000

# Full training with self-play
python train.py --mode full --episodes 5000 --self-play
```

---

## 🎮 Game Mechanics

### Drone Attributes

| Attribute | Description | Range |
|-----------|-------------|-------|
| ❤️ HP | Health points | 0-100 |
| 🛡️ Shield | Regenerating protection | 0-50 |
| ⚡ Energy | Powers abilities | 0-100 |
| 🔫 Ammo | Machine gun rounds | 0-500 |
| 🚀 Missiles | Homing missiles | 0-4 |

### Controls (Human vs AI Mode)

| Key | Action |
|-----|--------|
| `W/S` | Throttle up/down |
| `A/D` | Turn left/right |
| `Q/E` | Roll left/right |
| `↑/↓` | Pitch up/down |
| `Space` | Fire machine gun |
| `F` | Fire missile |
| `Shift` | Boost |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SkyBattle Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      Frontend (Vue 3 + Three.js)             │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │ WebSocket / REST                      │
│  ┌──────────────────────────▼──────────────────────────────────┐   │
│  │                      Backend (FastAPI)                       │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│  ┌──────────────────────────▼──────────────────────────────────┐   │
│  │           Core (Combat Env + MAPPO + Physics)                │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
SkyBattle/
├── backend/                    # Python backend
│   ├── envs/                   # Gymnasium environments
│   ├── agents/                 # RL algorithms (MAPPO)
│   ├── api/                    # FastAPI server
│   └── game/                   # Game logic
├── frontend/                   # Vue 3 + Three.js frontend
├── models/                     # Pretrained models
├── configs/                    # Configuration files
├── train.py                    # Training entry point
├── server.py                   # Server entry point
└── requirements.txt            # Python dependencies
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**✈️ Ready for takeoff? Let's battle! ✈️**

</div>
