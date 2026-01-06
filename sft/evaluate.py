"""
Evaluation script for trained models.
Loads checkpoints and evaluates on validation data.
"""

import json
import asyncio
import os
from typing import List, Dict
import argparse

import tinker
from dotenv import load_dotenv

from . import config
from .data_loader import load_dataset, transform_to_target_format, split_dataset
from .prompt_builder import build_instruction_prompt


async def generate_sample(sampling_client, tokenizer, user_query: str, max_tokens: int = 256) -> str:
    """
    Generate a response for a given user query.
    
    Args:
        sampling_client: Tinker sampling client
        tokenizer: Tokenizer instance
        user_query: User's query
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated response string
    """
    # Build prompt
    instruction_prompt, _ = build_instruction_prompt(user_query, "")
    
    # Tokenize (without target)
    tokens = tokenizer.encode(instruction_prompt, add_special_tokens=False)
    
    # Create model input
    model_input = tinker.types.ModelInput.from_ints(tokens)
    
    # Sampling parameters
    sampling_params = tinker.types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,  # Greedy decoding for evaluation
        top_p=1.0
    )
    
    # Sample
    sample_future = await sampling_client.sample_async(model_input, sampling_params=sampling_params, num_samples=1)
    result = await sample_future
    
    # Decode generated tokens
    generated_tokens = result.sequences[0].tokens
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text


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
        # Sometimes model generates extra text
        start_idx = output_str.find('{')
        end_idx = output_str.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            return None
        
        json_str = output_str[start_idx:end_idx + 1]
        parsed = json.loads(json_str)
        
        # Validate structure
        if 'final' in parsed:
            # Direct answer format
            return parsed
        elif 'tool' in parsed and 'args' in parsed:
            # Tool call format
            if parsed['tool'] in ['calculator', 'python']:
                return parsed
        
        return None
    except json.JSONDecodeError:
        return None


def calculate_metrics(predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts
        
    Returns:
        Dict with evaluation metrics
    """
    total = len(predictions)
    valid_json = sum(1 for p in predictions if p['parsed'] is not None)
    
    # Calculate accuracy for valid JSONs
    correct_tool = 0
    correct_format = 0
    
    for pred, gt in zip(predictions, ground_truths):
        if pred['parsed'] is None:
            continue
        
        # Check if correct format
        gt_json = json.loads(gt['expected_output'])
        
        if 'final' in gt_json and 'final' in pred['parsed']:
            correct_format += 1
        elif 'tool' in gt_json and 'tool' in pred['parsed']:
            correct_format += 1
            if gt_json['tool'] == pred['parsed']['tool']:
                correct_tool += 1
    
    return {
        'total_samples': total,
        'valid_json_count': valid_json,
        'valid_json_rate': valid_json / total if total > 0 else 0,
        'correct_format_count': correct_format,
        'correct_format_rate': correct_format / total if total > 0 else 0,
        'correct_tool_count': correct_tool,
        'correct_tool_rate': correct_tool / total if total > 0 else 0
    }


async def evaluate(checkpoint_name: str, num_samples: int = 100):
    """
    Main evaluation function.
    
    Args:
        checkpoint_name: Name of the checkpoint to evaluate
        num_samples: Number of samples to evaluate
    """
    print("=" * 80)
    print(f"Evaluating Checkpoint: {checkpoint_name}")
    print("=" * 80)
    
    # Load environment variables
    load_dotenv()
    tinker_api_key = os.getenv("TINKER_API_KEY")
    if not tinker_api_key:
        raise ValueError("TINKER_API_KEY environment variable is not set. Please create a .env file with your API key.")
    
    # Initialize Tinker client
    print(f"\nInitializing Tinker client...")
    service_client = tinker.ServiceClient(api_key=tinker_api_key)
    
    # Load checkpoint as sampling client
    print(f"Loading checkpoint: {checkpoint_name}")
    sampling_client = await service_client.create_lora_sampling_client_async(
        base_model=config.BASE_MODEL,
        lora_checkpoint=checkpoint_name
    )
    print("Checkpoint loaded")
    
    # Get tokenizer from Tinker client
    print(f"\nGetting tokenizer from Tinker client...")
    tokenizer = sampling_client.get_tokenizer()
    print("Tokenizer initialized")
    
    # Load validation data
    print(f"\nLoading validation data...")
    data = load_dataset(config.DATASET_PATH)
    _, val_data = split_dataset(data, train_ratio=config.TRAIN_RATIO)
    
    # Limit number of samples
    eval_samples = val_data[:num_samples]
    print(f"Evaluating on {len(eval_samples)} samples")
    
    # Run evaluation
    print(f"\n{'=' * 80}")
    print("Running Evaluation")
    print(f"{'=' * 80}\n")
    
    predictions = []
    ground_truths = []
    
    for i, sample in enumerate(eval_samples):
        try:
            # Transform sample
            transformed = transform_to_target_format(sample)
            user_query = transformed['input']
            expected_output = transformed['output']
            
            # Generate prediction
            generated = await generate_sample(sampling_client, tokenizer, user_query)
            
            # Validate JSON
            parsed = validate_json_output(generated)
            
            predictions.append({
                'input': user_query,
                'generated': generated,
                'parsed': parsed
            })
            
            ground_truths.append({
                'input': user_query,
                'expected_output': expected_output
            })
            
            if (i + 1) % 10 == 0:
                print(f"Evaluated {i + 1}/{len(eval_samples)} samples")
                
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            predictions.append({
                'input': user_query,
                'generated': None,
                'parsed': None,
                'error': str(e)
            })
            ground_truths.append({
                'input': user_query,
                'expected_output': expected_output
            })
    
    # Calculate metrics
    print(f"\n{'=' * 80}")
    print("Evaluation Results")
    print(f"{'=' * 80}\n")
    
    metrics = calculate_metrics(predictions, ground_truths)
    
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Valid JSON rate: {metrics['valid_json_rate']:.2%} ({metrics['valid_json_count']}/{metrics['total_samples']})")
    print(f"Correct format rate: {metrics['correct_format_rate']:.2%} ({metrics['correct_format_count']}/{metrics['total_samples']})")
    print(f"Correct tool rate: {metrics['correct_tool_rate']:.2%} ({metrics['correct_tool_count']}/{metrics['total_samples']})")
    
    # Show some examples
    print(f"\n{'=' * 80}")
    print("Sample Predictions")
    print(f"{'=' * 80}\n")
    
    for i in range(min(5, len(predictions))):
        print(f"Example {i + 1}:")
        print(f"  Input: {predictions[i]['input'][:100]}...")
        print(f"  Generated: {predictions[i]['generated']}")
        print(f"  Expected: {ground_truths[i]['expected_output']}")
        print(f"  Valid: {predictions[i]['parsed'] is not None}")
        print()
    
    return metrics


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained model checkpoint")
    parser.add_argument("checkpoint_name", type=str, help="Name of the checkpoint to evaluate")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate (default: 100)")
    
    args = parser.parse_args()
    
    asyncio.run(evaluate(args.checkpoint_name, args.num_samples))


if __name__ == "__main__":
    main()

