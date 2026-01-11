# Contributing to SkyBattle

Thank you for your interest in contributing to SkyBattle! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

---

## 📜 Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git
- (Optional) CUDA 12.x for GPU training

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/SkyBattle.git
cd SkyBattle
git remote add upstream https://github.com/YYYJH1/SkyBattle.git
```

---

## 💻 Development Setup

### Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run only backend
docker-compose up backend

# Run with GPU training
docker-compose --profile training up
```

---

## 🤝 How to Contribute

### Types of Contributions

1. **🐛 Bug Reports** - Report bugs in the [issue tracker](https://github.com/YYYJH1/SkyBattle/issues)
2. **💡 Feature Requests** - Suggest new features via issues
3. **📝 Documentation** - Improve docs, fix typos, add examples
4. **🔧 Code** - Fix bugs, implement features, improve performance
5. **🧪 Tests** - Add test cases, improve test coverage
6. **🎨 UI/UX** - Improve the frontend interface

### Issue Guidelines

Before creating an issue:

1. Search existing issues to avoid duplicates
2. Use issue templates when available
3. Provide detailed information:
   - For bugs: steps to reproduce, expected vs actual behavior
   - For features: use case, proposed implementation

---

## 🔄 Pull Request Process

### 1. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 2. Make Changes

- Follow the [Coding Standards](#coding-standards)
- Add tests for new functionality
- Update documentation if needed

### 3. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new algorithm support"
```

#### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(env): add new obstacle system
fix(agent): resolve memory leak in buffer
docs(readme): update installation instructions
```

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

### 5. PR Checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Code is formatted (`black backend/`, `ruff check backend/`)
- [ ] Documentation updated if needed
- [ ] Commit messages follow convention
- [ ] PR description explains the changes

---

## 📏 Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

```bash
# Format code
black backend/ --line-length 100

# Check linting
ruff check backend/

# Type checking (optional)
mypy backend/
```

#### Key Guidelines

```python
# Good: Descriptive names
def calculate_reward(damage_dealt: float, kills: int) -> float:
    """Calculate agent reward based on combat performance."""
    return damage_dealt * 0.5 + kills * 50.0

# Good: Type hints
def create_drone(
    drone_id: str,
    team: str,
    position: Optional[np.ndarray] = None,
) -> Drone:
    ...

# Good: Docstrings for public functions
def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
    """
    Reset the environment to initial state.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        observations: Dict of agent observations
        info: Additional information
    """
    ...
```

### TypeScript/Vue Style

```bash
# In frontend/
npm run lint
npm run format
```

---

## 🧪 Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_env.py -v

# With coverage
pytest tests/ --cov=backend --cov-report=html
```

### Writing Tests

```python
# tests/test_example.py

import pytest
from backend.envs import CombatEnv

class TestCombatEnv:
    """Tests for CombatEnv."""
    
    def test_reset_returns_correct_shape(self):
        """Test that reset returns observations with correct shape."""
        env = CombatEnv()
        obs, info = env.reset()
        
        assert len(obs) == env.n_agents
        for agent_obs in obs.values():
            assert agent_obs.shape == (env.obs_dim,)
    
    @pytest.mark.parametrize("team_size", [2, 3, 5])
    def test_various_team_sizes(self, team_size):
        """Test environment with different team sizes."""
        env = CombatEnv(config=CombatConfig(team_size=team_size))
        assert env.n_agents == team_size * 2
```

---

## 📚 Documentation

### Updating Documentation

- `README.md` - Project overview
- `docs/REQUIREMENTS.md` - Detailed requirements
- `docs/API.md` - API documentation
- Inline docstrings for code

### Docstring Format

```python
def function_name(arg1: Type1, arg2: Type2) -> ReturnType:
    """
    Brief description of the function.
    
    Longer description if needed, explaining the function's
    behavior in more detail.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When this exception is raised
        
    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
    """
    ...
```

---

## ❓ Questions?

If you have questions:

1. Check existing [issues](https://github.com/YYYJH1/SkyBattle/issues)
2. Read the [documentation](docs/)
3. Open a new issue with the `question` label

---

## 🙏 Thank You!

Thank you for contributing to SkyBattle! Your contributions help make this project better for everyone.

---

<div align="center">

**Happy Coding! ✈️**

</div>
