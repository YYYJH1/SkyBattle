"""Logging utilities with TensorBoard support."""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class TensorBoardLogger:
    """TensorBoard logger for training metrics."""
    
    def __init__(self, log_dir: str = "runs", experiment_name: Optional[str] = None):
        if not HAS_TENSORBOARD:
            raise ImportError("TensorBoard not installed. Run: pip install tensorboard")
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.step = 0
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a scalar value."""
        step = step if step is not None else self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: Optional[int] = None):
        """Log multiple scalars."""
        step = step if step is not None else self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: Optional[int] = None):
        """Log a histogram."""
        step = step if step is not None else self.step
        self.writer.add_histogram(tag, values, step)
    
    def log_episode(self, episode: int, metrics: Dict[str, float]):
        """Log episode-level metrics."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"episode/{key}", value, episode)
    
    def log_training(self, step: int, metrics: Dict[str, float]):
        """Log training step metrics."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"training/{key}", value, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Optional[Dict[str, float]] = None):
        """Log hyperparameters."""
        metrics = metrics or {}
        self.writer.add_hparams(hparams, metrics)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text."""
        step = step if step is not None else self.step
        self.writer.add_text(tag, text, step)
    
    def increment_step(self):
        """Increment the global step counter."""
        self.step += 1
    
    def flush(self):
        """Flush the writer."""
        self.writer.flush()
    
    def close(self):
        """Close the writer."""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ConsoleLogger:
    """Simple console logger for training progress."""
    
    def __init__(self, log_dir: Optional[str] = None, log_to_file: bool = False):
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_to_file = log_to_file
        
        if self.log_to_file and self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / "training.log"
        else:
            self.log_file = None
        
        self.metrics_history = []
    
    def log(self, message: str):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(formatted + "\n")
    
    def log_episode(self, episode: int, metrics: Dict[str, float]):
        """Log episode metrics."""
        self.metrics_history.append({"episode": episode, **metrics})
        
        parts = [f"Episode {episode}"]
        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}: {value:.3f}")
            else:
                parts.append(f"{key}: {value}")
        
        self.log(" | ".join(parts))
    
    def log_training(self, step: int, metrics: Dict[str, float]):
        """Log training step."""
        parts = [f"Step {step}"]
        for key, value in metrics.items():
            parts.append(f"{key}: {value:.4f}")
        
        self.log(" | ".join(parts))
    
    def save_metrics(self, path: Optional[str] = None):
        """Save metrics history to JSON."""
        if path is None and self.log_dir:
            path = str(self.log_dir / "metrics.json")
        elif path is None:
            path = "metrics.json"
        
        with open(path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of training."""
        if not self.metrics_history:
            return {}
        
        # Get last 100 episodes for averages
        recent = self.metrics_history[-100:]
        
        summary = {}
        keys = [k for k in recent[0].keys() if k != "episode"]
        
        for key in keys:
            values = [m[key] for m in recent if key in m and isinstance(m[key], (int, float))]
            if values:
                summary[f"mean_{key}"] = sum(values) / len(values)
                summary[f"max_{key}"] = max(values)
                summary[f"min_{key}"] = min(values)
        
        return summary
