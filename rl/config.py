"""
Configuration module for RL training.
Contains hyperparameters and settings for PPO training pipeline.
"""

# Model Configuration
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SFT_CHECKPOINT = "tinker://ef65351a-3e46-5958-b6eb-c6f8e922b546:train:0/weights/checkpoint_step_600"  # From SFT training
LORA_RANK = 32

# RL Hyperparameters
RL_LEARNING_RATE = 1e-5  # Lower than SFT for stability
BATCH_SIZE = 8
NUM_STEPS = 500
MAX_TOKENS = 128  # Max generation length for tool calling outputs

# Sampling Configuration
TEMPERATURE = 1.0  # For exploration during RL
TOP_P = 1.0
SAMPLER_REFRESH_INTERVAL = 10  # Refresh sampler every N steps

# Data Configuration
DATASET_PATH = "data/tool_calling_sft_dataset.json"
TRAIN_RATIO = 0.8

# Reward Configuration (for reference, actual values in reward.py)
REWARD_FORMAT_VALID = 1.0
REWARD_FORMAT_INVALID = 0.0
REWARD_TOOL_CORRECT = 0.5
REWARD_TOOL_INCORRECT = -0.5
REWARD_ARGS_CORRECT = 1.0
REWARD_ARGS_INCORRECT = 0.0
REWARD_LENGTH_PENALTY = -0.2

# Logging Configuration
LOG_DIR = "logs"
LOG_INTERVAL = 10  # Log metrics every N steps
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N steps

