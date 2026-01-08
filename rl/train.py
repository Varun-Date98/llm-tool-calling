"""
Main RL training script using Tinker API with PPO.
Trains the model to improve tool selection and argument accuracy.
"""

import asyncio
import os
import json
import random
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np

import tinker
from tinker import types as tinker_types
from dotenv import load_dotenv

from . import config
from .reward import compute_reward, compute_batch_rewards
from .metrics import MetricsLogger, StepMetrics, compute_aggregate_metrics

# Import from SFT module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sft.data_loader import load_dataset, transform_to_target_format, split_dataset
from sft.prompt_builder import build_instruction_prompt


async def sample_batch(
    sampler,
    tokenizer,
    batch_prompts: List[str],
    max_tokens: int = 128,
    temperature: float = 1.0
) -> List[Tuple[List[int], List[float], str]]:
    """
    Sample outputs from the model for a batch of prompts.
    
    Args:
        sampler: Tinker sampling client
        tokenizer: Tokenizer instance
        batch_prompts: List of instruction prompts
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        List of tuples: (generated_tokens, logprobs, decoded_text)
    """
    # Prepare model inputs
    prompt_inputs = []
    prompt_token_lists = []
    
    for prompt in batch_prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_inputs.append(tinker_types.ModelInput.from_ints(tokens))
        prompt_token_lists.append(tokens)
    
    # Sampling parameters
    sampling_params = tinker_types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=config.TOP_P,
        stop=["}"]  # Stop at end of JSON
    )
    
    # Fire all sample requests in parallel
    sample_futures = [
        sampler.sample_async(
            prompt=p_input,
            sampling_params=sampling_params,
            num_samples=1,
        )
        for p_input in prompt_inputs
    ]
    
    # Gather results
    sample_results = await asyncio.gather(*sample_futures)
    
    # Extract tokens, logprobs, and decoded text
    outputs = []
    for i, result in enumerate(sample_results):
        seq = result.sequences[0]
        gen_tokens = list(seq.tokens)
        gen_logprobs = list(seq.logprobs) if seq.logprobs else [0.0] * len(gen_tokens)
        decoded_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
        outputs.append((
            prompt_token_lists[i],  # prompt tokens
            gen_tokens,              # generated tokens
            gen_logprobs,            # logprobs for generated tokens
            decoded_text             # decoded generated text
        ))
    
    return outputs


async def compute_kl_divergence(
    ref_client,
    tokenizer,
    prompt_tokens_list: List[List[int]],
    action_tokens_list: List[List[int]],
    policy_logprobs_list: List[List[float]]
) -> Tuple[float, List[float]]:
    """
    Compute KL divergence between current policy and reference (SFT) model.
    
    KL(policy || ref) = E[log(policy) - log(ref)]
    
    Args:
        ref_client: Reference model training client
        tokenizer: Tokenizer instance
        prompt_tokens_list: List of prompt token sequences
        action_tokens_list: List of generated action token sequences
        policy_logprobs_list: List of policy logprobs for actions
        
    Returns:
        Tuple of (average_kl, per_sample_kl_list)
    """
    # Create datums for reference model forward pass
    ref_datums = []
    
    for prompt_tokens, action_tokens in zip(prompt_tokens_list, action_tokens_list):
        # Full sequence: prompt + action
        full_tokens = prompt_tokens + action_tokens
        
        # Weights: 0 for prompt, 1 for action (we only care about action logprobs)
        weights = [0.0] * len(prompt_tokens) + [1.0] * len(action_tokens)
        
        datum = tinker_types.Datum(
            model_input=tinker_types.ModelInput.from_ints(full_tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": np.array(full_tokens[1:], dtype=np.int64),
                "weights": np.array(weights[1:], dtype=np.float32),
            },
        )
        ref_datums.append(datum)
    
    # Forward pass on reference model (no gradients needed, just logprobs)
    ref_future = await ref_client.forward_backward_async(ref_datums, loss_fn="cross_entropy")
    ref_result = await ref_future
    
    # Compute KL for each sample
    per_sample_kl = []
    
    for i, (prompt_tokens, action_tokens, policy_logprobs) in enumerate(
        zip(prompt_tokens_list, action_tokens_list, policy_logprobs_list)
    ):
        # Get reference logprobs from result
        ref_output = ref_result.loss_fn_outputs[i]
        ref_logprobs = np.array(ref_output["logprobs"].to_numpy(), dtype=np.float32)
        
        # Extract logprobs for action tokens only (skip prompt)
        prompt_len = len(prompt_tokens)
        ref_action_logprobs = ref_logprobs[prompt_len - 1:]  # -1 due to shift
        
        # Ensure lengths match
        min_len = min(len(policy_logprobs), len(ref_action_logprobs))
        if min_len > 0:
            policy_lp = np.array(policy_logprobs[:min_len])
            ref_lp = ref_action_logprobs[:min_len]
            
            # KL divergence: KL(policy || ref) = sum(policy_logprob - ref_logprob)
            # This represents E_policy[log(policy/ref)]
            # However, this can be negative. To ensure non-negativity, we compute:
            # KL = max(0, sum(policy_lp - ref_lp))
            # This ensures KL is always >= 0
            kl_raw = float(np.sum(policy_lp - ref_lp))
            kl = max(0.0, kl_raw)  # Ensure non-negativity
            
            # Alternative: Use absolute value if we want to track divergence in both directions
            # But standard KL divergence should be non-negative
            per_sample_kl.append(kl)
        else:
            per_sample_kl.append(0.0)
    
    # Average KL across batch
    avg_kl = sum(per_sample_kl) / len(per_sample_kl) if per_sample_kl else 0.0
    
    return avg_kl, per_sample_kl


def create_ppo_datums(
    prompt_tokens_list: List[List[int]],
    action_tokens_list: List[List[int]],
    action_logprobs_list: List[List[float]],
    rewards: List[float]
) -> List[tinker_types.Datum]:
    """
    Create PPO datums for training.
    
    Following the Tinker PPO pattern from the reference implementation.
    
    Args:
        prompt_tokens_list: List of prompt token sequences
        action_tokens_list: List of action (generated) token sequences
        action_logprobs_list: List of logprobs for actions (pi_old)
        rewards: List of rewards (advantages) for each sample
        
    Returns:
        List of Tinker Datum objects for PPO training
    """
    ppo_datums = []
    
    for prompt_tokens, action_tokens, action_logprobs, reward in zip(
        prompt_tokens_list, action_tokens_list, action_logprobs_list, rewards
    ):
        # Full sequence: prompt + action
        full_tokens = prompt_tokens + action_tokens
        
        if len(full_tokens) < 2:
            continue
        
        # Input tokens (all except last)
        input_tokens = full_tokens[:-1]
        
        # Target tokens (all except first, shifted)
        target_tokens = full_tokens[1:]
        
        # Logprobs: 0 for prompt positions, actual logprobs for action
        prompt_len = len(prompt_tokens)
        all_logprobs = [0.0] * (prompt_len - 1) + list(action_logprobs)
        
        # Pad or truncate to match target length
        while len(all_logprobs) < len(target_tokens):
            all_logprobs.append(0.0)
        all_logprobs = all_logprobs[:len(target_tokens)]
        
        # Advantages: 0 for prompt positions, reward for action positions
        # Using constant reward across all action tokens (REINFORCE style)
        action_len = len(action_tokens)
        all_advantages = [0.0] * (prompt_len - 1) + [reward] * action_len
        
        # Pad or truncate to match target length
        while len(all_advantages) < len(target_tokens):
            all_advantages.append(0.0)
        all_advantages = all_advantages[:len(target_tokens)]
        
        datum = tinker_types.Datum(
            model_input=tinker_types.ModelInput.from_ints(input_tokens),
            loss_fn_inputs={
                "target_tokens": np.array(target_tokens, dtype=np.int64),
                "logprobs": np.array(all_logprobs, dtype=np.float32),
                "advantages": np.array(all_advantages, dtype=np.float32),
            },
        )
        ppo_datums.append(datum)
    
    return ppo_datums


def prepare_batch(
    train_data: List[Dict],
    batch_size: int
) -> List[Tuple[str, Dict]]:
    """
    Prepare a batch of training samples.
    
    Args:
        train_data: Training dataset
        batch_size: Number of samples per batch
        
    Returns:
        List of (instruction_prompt, expected_output_dict) tuples
    """
    # Sample batch from training data
    batch_samples = random.sample(train_data, min(batch_size, len(train_data)))
    
    batch = []
    for sample in batch_samples:
        try:
            transformed = transform_to_target_format(sample)
            user_query = transformed['input']
            expected_output_str = transformed['output']
            expected_output = json.loads(expected_output_str)
            
            # Build instruction prompt
            instruction_prompt, _ = build_instruction_prompt(user_query, "")
            
            batch.append((instruction_prompt, expected_output))
        except Exception as e:
            print(f"Warning: Failed to prepare sample: {e}")
            continue
    
    return batch


async def rl_train_step(
    rl_client,
    ref_client,
    sampler,
    tokenizer,
    batch: List[Tuple[str, Dict]],
    step: int
) -> Tuple[List[Dict], float]:
    """
    Execute one RL training step.
    
    Args:
        rl_client: Tinker RL training client
        ref_client: Reference model client (frozen SFT)
        sampler: Tinker sampling client
        tokenizer: Tokenizer instance
        batch: List of (prompt, expected_output) tuples
        step: Current step number
        
    Returns:
        Tuple of (rewards_list, kl_divergence)
    """
    prompts = [p for p, _ in batch]
    expected_outputs = [e for _, e in batch]
    
    # 1. Sample from current policy
    sample_outputs = await sample_batch(
        sampler,
        tokenizer,
        prompts,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE
    )
    
    # 2. Extract components
    prompt_tokens_list = []
    action_tokens_list = []
    action_logprobs_list = []
    generated_texts = []
    
    for prompt_tokens, gen_tokens, gen_logprobs, decoded_text in sample_outputs:
        prompt_tokens_list.append(prompt_tokens)
        action_tokens_list.append(gen_tokens)
        action_logprobs_list.append(gen_logprobs)
        generated_texts.append(decoded_text)
    
    # 3. Compute rewards
    rewards = []
    for gen_text, expected in zip(generated_texts, expected_outputs):
        reward_dict = compute_reward(gen_text, expected)
        rewards.append(reward_dict)
    
    # 4. Compute KL divergence against reference model
    kl_divergence, per_sample_kl = await compute_kl_divergence(
        ref_client,
        tokenizer,
        prompt_tokens_list,
        action_tokens_list,
        action_logprobs_list
    )
    
    # 5. Create PPO datums
    reward_values = [r['total'] for r in rewards]
    ppo_datums = create_ppo_datums(
        prompt_tokens_list,
        action_tokens_list,
        action_logprobs_list,
        reward_values
    )
    
    if not ppo_datums:
        print(f"Warning: No valid PPO datums at step {step}")
        return rewards, kl_divergence
    
    # 6. PPO forward/backward
    ppo_future = await rl_client.forward_backward_async(ppo_datums, loss_fn="ppo")
    await ppo_future
    
    # 7. Optimizer step
    optim_future = await rl_client.optim_step_async(
        tinker_types.AdamParams(learning_rate=config.RL_LEARNING_RATE)
    )
    await optim_future
    
    return rewards, kl_divergence


async def main():
    """
    Main RL training loop.
    """
    print("=" * 80)
    print("Starting RL Training with Tinker API (PPO)")
    print("=" * 80)
    
    # Load environment variables
    load_dotenv()
    tinker_api_key = os.getenv("TINKER_API_KEY")
    if not tinker_api_key:
        raise ValueError("TINKER_API_KEY environment variable is not set.")
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(config.LOG_DIR)
    
    # Initialize Tinker service client
    print(f"\nInitializing Tinker clients...")
    print(f"SFT checkpoint: {config.SFT_CHECKPOINT}")
    
    service_client = tinker.ServiceClient(api_key=tinker_api_key)
    
    # Create RL training client (starts from SFT checkpoint)
    print(f"Creating RL training client from SFT checkpoint: {config.SFT_CHECKPOINT}")
    rl_client = await service_client.create_training_client_from_state_async(
        path=config.SFT_CHECKPOINT,
        user_metadata={"purpose": "rl_training"}
    )
    
    # Create reference client (frozen SFT model for KL computation)
    print("Creating reference client for KL computation...")
    ref_client = await service_client.create_training_client_from_state_async(
        path=config.SFT_CHECKPOINT,
        user_metadata={"purpose": "rl_reference"}
    )
    
    # Get tokenizer
    tokenizer = rl_client.get_tokenizer()
    print("Tokenizer initialized")
    
    # Create initial sampler
    print("Creating initial sampler...")
    sampler = await rl_client.save_weights_and_get_sampling_client_async(name="rl_step_0")
    
    # Load dataset
    print(f"\nLoading dataset from {config.DATASET_PATH}...")
    data = load_dataset(config.DATASET_PATH)
    train_data, val_data = split_dataset(data, train_ratio=config.TRAIN_RATIO)
    
    # Training configuration summary
    print(f"\nRL Training Configuration:")
    print(f"  - Learning rate: {config.RL_LEARNING_RATE}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Number of steps: {config.NUM_STEPS}")
    print(f"  - Max tokens: {config.MAX_TOKENS}")
    print(f"  - Temperature: {config.TEMPERATURE}")
    print(f"  - Sampler refresh interval: {config.SAMPLER_REFRESH_INTERVAL}")
    print(f"  - Log interval: {config.LOG_INTERVAL}")
    print(f"  - Checkpoint interval: {config.CHECKPOINT_INTERVAL}")
    
    # Training loop
    print(f"\n{'=' * 80}")
    print("Starting Training Loop")
    print(f"{'=' * 80}\n")
    
    for step in range(1, config.NUM_STEPS + 1):
        try:
            # Prepare batch
            batch = prepare_batch(train_data, config.BATCH_SIZE)
            
            if not batch:
                print(f"Warning: Empty batch at step {step}, skipping...")
                continue
            
            # Execute RL training step
            rewards, kl_divergence = await rl_train_step(
                rl_client,
                ref_client,
                sampler,
                tokenizer,
                batch,
                step
            )
            
            # Log metrics
            step_metrics = metrics_logger.log_step(step, rewards, kl_divergence)
            
            # Print progress
            if step % config.LOG_INTERVAL == 0:
                metrics_logger.print_step(step_metrics)
            
            # Refresh sampler periodically
            if step % config.SAMPLER_REFRESH_INTERVAL == 0:
                print(f"Refreshing sampler at step {step}...")
                sampler = await rl_client.save_weights_and_get_sampling_client_async(
                    name=f"rl_step_{step}"
                )
            
            # Save checkpoint
            if step % config.CHECKPOINT_INTERVAL == 0:
                checkpoint_name = f"rl_checkpoint_step_{step}"
                print(f"Saving checkpoint: {checkpoint_name}")
                await rl_client.save_state_async(name=checkpoint_name)
                
                # Also save metrics
                metrics_logger.save()
            
        except Exception as e:
            print(f"Error at step {step}: {e}")
    
    # Final checkpoint and metrics save
    print(f"\n{'=' * 80}")
    print("Training Complete!")
    print(f"{'=' * 80}\n")
    
    final_checkpoint_name = "rl_final_checkpoint"
    print(f"Saving final checkpoint: {final_checkpoint_name}")
    await rl_client.save_state_async(name=final_checkpoint_name)
    
    # Save final metrics
    log_file = metrics_logger.save()
    
    # Print summary
    summary = metrics_logger.get_summary()
    print(f"\nTraining Summary:")
    print(f"  - Total steps: {summary.get('total_steps', 0)}")
    print(f"  - Average reward: {summary.get('avg_reward', 0):.4f}")
    print(f"  - Final reward: {summary.get('final_reward', 0):.4f}")
    print(f"  - Average KL: {summary.get('avg_kl', 0):.4f}")
    print(f"  - Final valid JSON rate: {summary.get('final_valid_json_rate', 0):.1%}")
    print(f"  - Final correct tool rate: {summary.get('final_correct_tool_rate', 0):.1%}")
    print(f"  - Final correct args rate: {summary.get('final_correct_args_rate', 0):.1%}")
    
    print(f"\nMetrics saved to: {log_file}")
    print("\nRL training pipeline completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())

