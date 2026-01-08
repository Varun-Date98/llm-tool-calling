"""
Metrics logging module for RL training.
Tracks rewards, KL divergence, and accuracy metrics per step.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict, Optional


@dataclass
class StepMetrics:
    """Metrics tracked per training step."""
    step: int
    
    # Reward components (batch averages)
    avg_total_reward: float
    avg_format_reward: float
    avg_tool_reward: float
    avg_args_reward: float
    avg_length_penalty: float
    
    # KL divergence
    kl_divergence: float
    
    # Accuracy rates
    valid_json_rate: float
    correct_tool_rate: float
    correct_args_rate: float
    
    # Batch info
    batch_size: int
    
    # Optional: min/max rewards for monitoring
    min_reward: float = 0.0
    max_reward: float = 0.0


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode/sample."""
    total_reward: float
    format_reward: float
    tool_reward: float
    args_reward: float
    length_penalty: float
    valid_json: bool
    correct_tool: bool
    correct_args: bool
    generated_text: str = ""
    expected_output: str = ""


class MetricsLogger:
    """Logger for tracking and saving RL training metrics."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = log_dir
        self.metrics_history: List[Dict] = []
        self.episode_buffer: List[EpisodeMetrics] = []
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(log_dir, f"rl_training_log_{timestamp}.json")
    
    def log_episode(self, metrics: EpisodeMetrics):
        """
        Log metrics for a single episode/sample.
        
        Args:
            metrics: Episode metrics
        """
        self.episode_buffer.append(metrics)
    
    def log_step(self, step: int, rewards: List[Dict], kl_divergence: float):
        """
        Log metrics for a training step from batch of rewards.
        
        Args:
            step: Training step number
            rewards: List of reward dictionaries from compute_reward()
            kl_divergence: KL divergence for this step
        """
        if not rewards:
            return
        
        batch_size = len(rewards)
        
        # Compute averages
        avg_total = sum(r['total'] for r in rewards) / batch_size
        avg_format = sum(r['format_reward'] for r in rewards) / batch_size
        avg_tool = sum(r['tool_reward'] for r in rewards) / batch_size
        avg_args = sum(r['args_reward'] for r in rewards) / batch_size
        avg_length = sum(r['length_penalty'] for r in rewards) / batch_size
        
        # Compute rates
        valid_json_count = sum(1 for r in rewards if r['valid_json'])
        correct_tool_count = sum(1 for r in rewards if r['correct_tool'])
        correct_args_count = sum(1 for r in rewards if r['correct_args'])
        
        valid_json_rate = valid_json_count / batch_size
        correct_tool_rate = correct_tool_count / batch_size
        correct_args_rate = correct_args_count / batch_size
        
        # Min/max rewards
        min_reward = min(r['total'] for r in rewards)
        max_reward = max(r['total'] for r in rewards)
        
        # Create step metrics
        step_metrics = StepMetrics(
            step=step,
            avg_total_reward=avg_total,
            avg_format_reward=avg_format,
            avg_tool_reward=avg_tool,
            avg_args_reward=avg_args,
            avg_length_penalty=avg_length,
            kl_divergence=kl_divergence,
            valid_json_rate=valid_json_rate,
            correct_tool_rate=correct_tool_rate,
            correct_args_rate=correct_args_rate,
            batch_size=batch_size,
            min_reward=min_reward,
            max_reward=max_reward
        )
        
        self.metrics_history.append(asdict(step_metrics))
        
        return step_metrics
    
    def log_step_from_metrics(self, metrics: StepMetrics):
        """
        Log a pre-computed StepMetrics object.
        
        Args:
            metrics: StepMetrics object
        """
        self.metrics_history.append(asdict(metrics))
    
    def save(self, filename: Optional[str] = None):
        """
        Save metrics history to JSON file.
        
        Args:
            filename: Optional custom filename (uses default if None)
        """
        save_path = filename or self.log_filename
        
        with open(save_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"Metrics saved to: {save_path}")
        return save_path
    
    def get_latest_metrics(self) -> Optional[Dict]:
        """Get the most recent step metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics across all logged steps.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics_history:
            return {}
        
        # Compute averages across all steps
        num_steps = len(self.metrics_history)
        
        summary = {
            'total_steps': num_steps,
            'avg_reward': sum(m['avg_total_reward'] for m in self.metrics_history) / num_steps,
            'final_reward': self.metrics_history[-1]['avg_total_reward'],
            'avg_kl': sum(m['kl_divergence'] for m in self.metrics_history) / num_steps,
            'final_valid_json_rate': self.metrics_history[-1]['valid_json_rate'],
            'final_correct_tool_rate': self.metrics_history[-1]['correct_tool_rate'],
            'final_correct_args_rate': self.metrics_history[-1]['correct_args_rate'],
        }
        
        return summary
    
    def print_step(self, step_metrics: StepMetrics, verbose: bool = False):
        """
        Print step metrics to console.
        
        Args:
            step_metrics: Metrics to print
            verbose: If True, print all components
        """
        print(f"Step {step_metrics.step}: "
              f"Reward={step_metrics.avg_total_reward:.3f} "
              f"(fmt={step_metrics.avg_format_reward:.2f}, "
              f"tool={step_metrics.avg_tool_reward:.2f}, "
              f"args={step_metrics.avg_args_reward:.2f}, "
              f"len={step_metrics.avg_length_penalty:.2f}) | "
              f"KL={step_metrics.kl_divergence:.4f} | "
              f"JSON={step_metrics.valid_json_rate:.1%} "
              f"Tool={step_metrics.correct_tool_rate:.1%} "
              f"Args={step_metrics.correct_args_rate:.1%}")


def compute_aggregate_metrics(rewards: List[Dict], kl: float, step: int) -> StepMetrics:
    """
    Utility function to compute StepMetrics from a list of reward dicts.
    
    Args:
        rewards: List of reward dictionaries
        kl: KL divergence value
        step: Current step number
        
    Returns:
        StepMetrics object
    """
    batch_size = len(rewards)
    
    if batch_size == 0:
        return StepMetrics(
            step=step,
            avg_total_reward=0.0,
            avg_format_reward=0.0,
            avg_tool_reward=0.0,
            avg_args_reward=0.0,
            avg_length_penalty=0.0,
            kl_divergence=kl,
            valid_json_rate=0.0,
            correct_tool_rate=0.0,
            correct_args_rate=0.0,
            batch_size=0,
            min_reward=0.0,
            max_reward=0.0
        )
    
    return StepMetrics(
        step=step,
        avg_total_reward=sum(r['total'] for r in rewards) / batch_size,
        avg_format_reward=sum(r['format_reward'] for r in rewards) / batch_size,
        avg_tool_reward=sum(r['tool_reward'] for r in rewards) / batch_size,
        avg_args_reward=sum(r['args_reward'] for r in rewards) / batch_size,
        avg_length_penalty=sum(r['length_penalty'] for r in rewards) / batch_size,
        kl_divergence=kl,
        valid_json_rate=sum(1 for r in rewards if r['valid_json']) / batch_size,
        correct_tool_rate=sum(1 for r in rewards if r['correct_tool']) / batch_size,
        correct_args_rate=sum(1 for r in rewards if r['correct_args']) / batch_size,
        batch_size=batch_size,
        min_reward=min(r['total'] for r in rewards),
        max_reward=max(r['total'] for r in rewards)
    )

