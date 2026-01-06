"""
Configuration module for SFT training.
Contains hyperparameters and settings for the training pipeline.
"""

# Model Configuration
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LORA_RANK = 32  # Default LoRA rank

# Training Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 5e-5  # Adam optimizer learning rate
NUM_STEPS = 800  # Total number of training steps
MAX_LENGTH = 512
GRADIENT_ACCUMULATION_STEPS = 1  # Number of steps to accumulate gradients

# Data Configuration
DATASET_PATH = "data/tool_calling_sft_dataset.json"
TRAIN_RATIO = 0.8  # 80% train, 20% validation

# Checkpoint Configuration
CHECKPOINT_INTERVAL = 200  # Save checkpoint every N steps
LOG_INTERVAL = 50  # Log metrics every N steps
VALIDATION_INTERVAL = 200  # Run validation every N steps
VALIDATION_SAMPLES = 50  # Number of samples to validate on

# Logging Configuration
LOG_DIR = "logs"
WANDB_ENABLED = False  # Set to True to enable Weights & Biases logging
WANDB_PROJECT = "tool-calling-sft"

