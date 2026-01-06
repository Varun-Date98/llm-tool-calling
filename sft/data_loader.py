"""
Data loading and transformation module.
Handles loading the dataset and converting samples to target JSON format.
"""

import json
import random
from typing import Dict, List, Tuple


def load_dataset(dataset_path: str) -> List[Dict]:
    """
    Load JSON dataset from file.
    
    Args:
        dataset_path: Path to the JSON dataset file
        
    Returns:
        List of dataset samples
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples from {dataset_path}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in dataset file: {e}")


def transform_to_target_format(sample: Dict) -> Dict[str, str]:
    """
    Convert dataset sample to target JSON output format.
    
    Args:
        sample: Dataset sample with fields:
            - prompt: User query
            - tool_needed: Boolean indicating if tool is needed
            - tool: Tool name (calculator/python) or None
            - tool_args: Tool arguments or None
            - expected_answer: Expected answer
            
    Returns:
        Dict with 'input' (user prompt) and 'output' (target JSON string)
    """
    user_prompt = sample['prompt']
    
    # Create target JSON based on tool requirement
    if not sample['tool_needed']:
        # No tool needed - direct answer
        target_json = {
            "final": sample['expected_answer']
        }
    elif sample['tool'] == 'calculator':
        # Calculator tool
        target_json = {
            "tool": "calculator",
            "args": {
                "expression": sample['tool_args']['expression']
            }
        }
    elif sample['tool'] == 'python':
        # Python execution tool
        target_json = {
            "tool": "python",
            "args": {
                "code": sample['tool_args']['code']
            }
        }
    else:
        raise ValueError(f"Unknown tool type: {sample.get('tool')}")
    
    # Convert to JSON string
    target_json_str = json.dumps(target_json, ensure_ascii=False)
    
    return {
        'input': user_prompt,
        'output': target_json_str
    }


def split_dataset(data: List[Dict], train_ratio: float = 0.9, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into training and validation sets.
    
    Args:
        data: List of dataset samples
        train_ratio: Ratio of training samples (default: 0.9)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data)
    """
    # Shuffle data with fixed seed for reproducibility
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Split
    split_idx = int(len(shuffled_data) * train_ratio)
    train_data = shuffled_data[:split_idx]
    val_data = shuffled_data[split_idx:]
    
    print(f"Dataset split: {len(train_data)} training samples, {len(val_data)} validation samples")
    
    return train_data, val_data


def validate_dataset(data: List[Dict]) -> bool:
    """
    Validate dataset structure.
    
    Args:
        data: List of dataset samples
        
    Returns:
        True if dataset is valid
        
    Raises:
        ValueError if dataset has issues
    """
    required_fields = ['prompt', 'tool_needed', 'tool', 'tool_args', 'expected_answer']
    
    for i, sample in enumerate(data):
        # Check required fields
        for field in required_fields:
            if field not in sample:
                raise ValueError(f"Sample {i} missing required field: {field}")
        
        # Validate tool-specific fields
        if sample['tool_needed']:
            if sample['tool'] not in ['calculator', 'python']:
                raise ValueError(f"Sample {i} has invalid tool: {sample['tool']}")
            if not sample['tool_args']:
                raise ValueError(f"Sample {i} has tool_needed=True but no tool_args")
        else:
            if sample['tool'] is not None:
                raise ValueError(f"Sample {i} has tool_needed=False but tool is not None")
    
    print(f"Dataset validation passed for {len(data)} samples")
    return True

