"""
Main training script for SFT using Tinker API with async/await.
"""

import asyncio
import os
import json
from datetime import datetime
from typing import List, Dict
import random
import numpy as np

import tinker
from dotenv import load_dotenv

from . import config
from .data_loader import load_dataset, transform_to_target_format, split_dataset, validate_dataset
from .prompt_builder import build_instruction_prompt
from .datum_builder import create_datum_with_truncation


def validate_json_output(output_str: str) -> Dict:
    """
    Validate that output is valid JSON and has correct structure.
    
    Args:
        output_str: Generated output string
        
    Returns:
        Parsed JSON object or None if invalid
    """
    try:
        # Try to find JSON in the output
        start_idx = output_str.find('{')
        end_idx = output_str.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            return None
        
        json_str = output_str[start_idx:end_idx + 1]
        parsed = json.loads(json_str)
        
        # Validate structure
        if 'final' in parsed:
            return parsed
        elif 'tool' in parsed and 'args' in parsed:
            if parsed['tool'] in ['calculator', 'python']:
                return parsed
        
        return None
    except (json.JSONDecodeError, KeyError):
        return None


async def train_step(training_client, batch_datums: List, step_num: int) -> Dict:
    """
    Execute one training step with a batch of data.
    
    Args:
        training_client: Tinker LoRA training client
        batch_datums: List of Tinker Datum objects
        step_num: Current step number
        
    Returns:
        Dict with training metrics
    """
    # Forward/backward pass for the entire batch at once
    # The API expects a list of Datum objects, not individual datums
    future_result = await training_client.forward_backward_async(batch_datums, loss_fn="cross_entropy")
    result = await future_result
    # Extract loss from ForwardBackwardOutput
    # The loss is stored in the metrics dictionary as 'loss:sum'
    supervised_tokens = 0.0
    loss_sum = result.metrics.get("loss:sum", 0.0)
    
    for d in batch_datums:
        w_np = np.array(d.loss_fn_inputs["weights"].to_numpy(), dtype=np.float32)
        supervised_tokens += float(np.sum(w_np))
    
    avg_loss = loss_sum / max(1.0,supervised_tokens)
    losses = [avg_loss]  # Single loss for the batch
    
    # Optimizer step
    # Use tinker's AdamParams with configured learning rate
    from tinker import types as tinker_types
    optim_future = training_client.optim_step_async(tinker_types.AdamParams(learning_rate=config.LEARNING_RATE))
    await optim_future
    
    return {
        'step': step_num,
        'loss': avg_loss,
        'batch_size': len(batch_datums),
        'losses': losses
    }


async def save_checkpoint(training_client, checkpoint_name: str):
    """
    Save model checkpoint asynchronously using Tinker's save_state API.
    
    Args:
        training_client: Tinker LoRA training client
        checkpoint_name: Name for the checkpoint
    """
    print(f"Saving checkpoint: {checkpoint_name}")
    try:
        # Use Tinker's save_state API
        save_future = training_client.save_state_async(name=checkpoint_name)
        result = await save_future
        print(f"Checkpoint saved successfully: {checkpoint_name}")
        return result
    except Exception as e:
        print(f"Warning: Failed to save checkpoint {checkpoint_name}: {e}")
        return None


def prepare_batch(train_data: List[Dict], batch_size: int, tokenizer, max_length: int) -> List:
    """
    Prepare a batch of training data.
    
    Args:
        train_data: Training dataset
        batch_size: Number of samples per batch
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        List of Datum objects
    """
    # Sample batch from training data
    batch_samples = random.sample(train_data, min(batch_size, len(train_data)))
    
    # Create datums
    batch_datums = []
    for sample_idx, sample in enumerate(batch_samples):
        try:
            # Transform to target format
            transformed = transform_to_target_format(sample)
            user_query = transformed['input']
            target_json = transformed['output']
            
            # Build instruction prompt
            instruction_prompt, target_json = build_instruction_prompt(user_query, target_json)
            
            # Create datum with truncation
            datum = create_datum_with_truncation(
                tokenizer,
                instruction_prompt,
                target_json,
                max_length=max_length
            )
            
            batch_datums.append(datum)
        except Exception as e:
            print(f"Warning: Failed to process sample: {e}")
            continue
    
    return batch_datums


async def run_validation(sampling_client, tokenizer, val_data: List[Dict], num_samples: int = 20) -> Dict:
    """
    Run validation on a subset of validation data using batched inference.
    
    Args:
        sampling_client: Tinker LoRA sampling client
        tokenizer: Tokenizer instance
        val_data: Validation dataset
        num_samples: Number of samples to validate on
        
    Returns:
        Dict with validation metrics
    """
    if not val_data:
        print(f"Warning: No validation data provided")
        return {'valid_json_rate': 0.0, 'correct_format_rate': 0.0, 'correct_tool_rate': 0.0}
    
    samples_to_validate = random.sample(val_data, min(num_samples, len(val_data)))
    
    # Step 1: Prepare all prompts and expected outputs
    prompt_inputs = []
    expected_outputs = []
    
    for sample in samples_to_validate:
        try:
            transformed = transform_to_target_format(sample)
            user_query = transformed['input']
            expected_output = transformed['output']
            
            instruction_prompt, _ = build_instruction_prompt(user_query, "")
            tokens = tokenizer.encode(instruction_prompt, add_special_tokens=False)
            model_input = tinker.types.ModelInput.from_ints(tokens)
            
            prompt_inputs.append(model_input)
            expected_outputs.append(expected_output)
        except Exception as e:
            print(f"Warning: Failed to prepare sample: {e}")
    
    if not prompt_inputs:
        print(f"Warning: No valid prompts prepared")
        return {'valid_json_rate': 0.0, 'correct_format_rate': 0.0, 'correct_tool_rate': 0.0}
    
    # Step 2: Fire all sample_async calls and gather results
    sampling_params = tinker.types.SamplingParams(
        max_tokens=128,
        temperature=0.0,  # Greedy decoding
        top_p=1.0,
        stop=["}"]
    )
    
    sample_futures = [
        sampling_client.sample_async(
            prompt=p_input,
            sampling_params=sampling_params,
            num_samples=1,
        )
        for p_input in prompt_inputs
    ]
    
    sample_results = await asyncio.gather(*sample_futures)
    
    # Step 3: Process results
    valid_json_count = 0
    correct_format_count = 0
    correct_tool_count = 0
    total_tool_count = 0
    
    for result, expected_output in zip(sample_results, expected_outputs):
        try:
            generated_tokens = result.sequences[0].tokens
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            parsed = validate_json_output(generated_text)
            
            if parsed is not None:
                valid_json_count += 1
                
                # Check if correct format
                expected_json = json.loads(expected_output)
                
                if 'final' in expected_json and 'final' in parsed:
                    correct_format_count += 1
                elif 'tool' in expected_json and 'tool' in parsed:
                    correct_format_count += 1
                    total_tool_count += 1
                    if expected_json['tool'] == parsed['tool']:
                        correct_tool_count += 1
        except Exception as e:
            print(f"Warning: Failed to validate sample: {e}")
    
    total = len(prompt_inputs)
    return {
        'valid_json_count': valid_json_count,
        'valid_json_rate': valid_json_count / total if total > 0 else 0.0,
        'correct_format_count': correct_format_count,
        'correct_format_rate': correct_format_count / total if total > 0 else 0.0,
        'correct_tool_count': correct_tool_count,
        'correct_tool_rate': correct_tool_count / total_tool_count if total_tool_count > 0 else 0.0,
        'total_samples': total,
        'total_tool_count': total_tool_count
    }


async def main():
    """
    Main training loop.
    """
    print("=" * 80)
    print("Starting SFT Training with Tinker API")
    print("=" * 80)
    
    # Load environment variables
    load_dotenv()
    tinker_api_key = os.getenv("TINKER_API_KEY")
    if not tinker_api_key:
        raise ValueError("TINKER_API_KEY environment variable is not set. Please create a .env file with your API key.")
    
    # Create log directory
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Initialize Tinker client
    print(f"\nInitializing Tinker client...")
    print(f"Base model: {config.BASE_MODEL}")
    print(f"LoRA rank: {config.LORA_RANK}")
    
    service_client = tinker.ServiceClient(api_key=tinker_api_key)
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.BASE_MODEL,
        rank=config.LORA_RANK
    )
    
    print("Tinker client initialized")
    
    # Get tokenizer from Tinker client
    print(f"\nGetting tokenizer from Tinker client...")
    tokenizer = training_client.get_tokenizer()
    print("Tokenizer initialized")
    
    # Load and prepare dataset
    print(f"\nLoading dataset from {config.DATASET_PATH}...")
    data = load_dataset(config.DATASET_PATH)
    validate_dataset(data)
    
    # Split dataset
    train_data, val_data = split_dataset(data, train_ratio=config.TRAIN_RATIO)
    
    # Training configuration
    print(f"\nTraining Configuration:")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Number of steps: {config.NUM_STEPS}")
    print(f"  - Max sequence length: {config.MAX_LENGTH}")
    print(f"  - Checkpoint interval: {config.CHECKPOINT_INTERVAL}")
    print(f"  - Log interval: {config.LOG_INTERVAL}")
    print(f"  - Validation interval: {config.VALIDATION_INTERVAL}")
    print(f"  - Validation samples: {config.VALIDATION_SAMPLES}")
    
    # Training loop
    print(f"\n{'=' * 80}")
    print("Starting Training Loop")
    print(f"{'=' * 80}\n")
    
    training_log = []
    
    for step in range(1, config.NUM_STEPS + 1):
        try:
            # Prepare batch
            batch_datums = prepare_batch(
                train_data,
                config.BATCH_SIZE,
                tokenizer,
                config.MAX_LENGTH
            )
            
            if not batch_datums:
                print(f"Warning: No valid datums in batch at step {step}, skipping...")
                continue
            
            # Execute training step
            metrics = await train_step(training_client, batch_datums, step)
            training_log.append(metrics)
            
            # Log metrics
            if step % config.LOG_INTERVAL == 0:
                print(f"Step {step}/{config.NUM_STEPS} | Loss: {metrics['loss']:.4f} | Batch size: {metrics['batch_size']}")
            
            # Run validation
            if (step < 200 and step % 50 == 0) or (step % config.VALIDATION_INTERVAL == 0):
                print(f"\nRunning validation at step {step}...")
                sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                    name=f"step_{step}"
                )
                val_metrics = await run_validation(
                    sampling_client, 
                    tokenizer, 
                    val_data, 
                    config.VALIDATION_SAMPLES
                )
                metrics['validation'] = val_metrics
                print(f"Validation Results:")
                print(f"  - Valid JSON: {val_metrics['valid_json_rate']:.1%} ({val_metrics['valid_json_count']}/{val_metrics['total_samples']})")
                print(f"  - Correct Format: {val_metrics['correct_format_rate']:.1%} ({val_metrics['correct_format_count']}/{val_metrics['total_samples']})")
                print(f"  - Correct Tool: {val_metrics['correct_tool_rate']:.1%} ({val_metrics['correct_tool_count']}/{val_metrics['total_tool_count']})")
                print()
            
            # Save checkpoint
            if step % config.CHECKPOINT_INTERVAL == 0:
                checkpoint_name = f"checkpoint_step_{step}"
                await save_checkpoint(training_client, checkpoint_name)
            
        except Exception as e:
            print(f"Error at step {step}: {e}")
            continue
    
    # Save final checkpoint
    print(f"\n{'=' * 80}")
    print("Training Complete!")
    print(f"{'=' * 80}\n")
    
    final_checkpoint_name = "final_checkpoint"
    await save_checkpoint(training_client, final_checkpoint_name)
    
    # Save training log
    log_file = os.path.join(config.LOG_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to: {log_file}")
    
    # Calculate and display final statistics
    if training_log:
        final_loss = training_log[-1]['loss']
        avg_loss = sum(log['loss'] for log in training_log) / len(training_log)
        print(f"\nFinal Statistics:")
        print(f"  - Final loss: {final_loss:.4f}")
        print(f"  - Average loss: {avg_loss:.4f}")
        print(f"  - Total steps completed: {len(training_log)}")
    
    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())

