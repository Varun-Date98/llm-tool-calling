"""
Datum creation module with weight masking.
Creates Tinker Datum objects following the compression project pattern.
"""

from tinker import types
import numpy as np
from typing import Any


def create_datum(tokenizer: Any, instruction_prompt: str, target_json: str) -> types.Datum:
    """
    Create Tinker Datum with proper weight masking.
    
    Tokenizes instruction_prompt and target_json separately, then concatenates them.
    Applies weight masking: 0 for instruction tokens, 1 for JSON output tokens.
    Uses language modeling shift pattern (t_input[:-1], t_input[1:]).
    
    Args:
        tokenizer: Hugging Face tokenizer (Qwen tokenizer)
        instruction_prompt: The instruction and user query text
        target_json: The target JSON output string
        
    Returns:
        Tinker Datum object with proper weight masking
    """
    # Tokenize separately to get exact token counts for accurate weight mask
    # Use add_special_tokens=False to match compression project pattern
    instruction_tokens = tokenizer.encode(instruction_prompt, add_special_tokens=False)
    json_tokens = tokenizer.encode(target_json, add_special_tokens=False)
    
    # Concatenate the tokenized sequences directly (same as compression project)
    t_input = instruction_tokens + json_tokens
    
    # Create weights using exact lengths from separate tokenization
    # 0 for instruction tokens, 1 for JSON output tokens
    t_weights = [0.0] * len(instruction_tokens) + [1.0] * len(json_tokens)
    
    # Create Datum using language modeling shift pattern (same as compression project):
    # - model_input: all tokens except last (tokens[0] to tokens[n-1])
    # - target_tokens: shifted by one (tokens[1] to tokens[n])
    # - weights: shifted accordingly
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(t_input[:-1]),
        loss_fn_inputs={
            "target_tokens": np.array(t_input[1:], dtype=np.int64),
            "weights": np.array(t_weights[1:], dtype=np.float32),
        },
    )
    
    return datum


def create_datum_batch(tokenizer: Any, batch_samples: list) -> list:
    """
    Create a batch of Datum objects.
    
    Args:
        tokenizer: Hugging Face tokenizer
        batch_samples: List of (instruction_prompt, target_json) tuples
        
    Returns:
        List of Tinker Datum objects
    """
    datums = []
    for instruction_prompt, target_json in batch_samples:
        try:
            datum = create_datum(tokenizer, instruction_prompt, target_json)
            datums.append(datum)
        except Exception as e:
            print(f"Warning: Failed to create datum for sample: {e}")
            continue
    
    return datums


def truncate_tokens(tokens: list, max_length: int) -> list:
    """
    Truncate token sequence to maximum length.
    
    Args:
        tokens: List of token IDs
        max_length: Maximum sequence length
        
    Returns:
        Truncated token list
    """
    if len(tokens) > max_length:
        return tokens[:max_length]
    return tokens


def create_datum_with_truncation(
    tokenizer: Any,
    instruction_prompt: str,
    target_json: str,
    max_length: int = 512
) -> types.Datum:
    """
    Create Datum with token truncation to respect max_length.
    
    Args:
        tokenizer: Hugging Face tokenizer
        instruction_prompt: The instruction and user query text
        target_json: The target JSON output string
        max_length: Maximum sequence length
        
    Returns:
        Tinker Datum object with truncated tokens
    """
    # Tokenize separately
    instruction_tokens = tokenizer.encode(instruction_prompt, add_special_tokens=False)
    json_tokens = tokenizer.encode(target_json, add_special_tokens=False)
    
    # Concatenate
    t_input = instruction_tokens + json_tokens
    
    # Truncate if necessary
    if len(t_input) > max_length:
        # Prefer keeping the end (JSON output) over the beginning
        # But ensure we have at least some instruction context
        min_instruction_len = min(50, len(instruction_tokens))
        if len(json_tokens) + min_instruction_len <= max_length:
            # Keep all JSON tokens and truncate instruction
            instruction_tokens = instruction_tokens[-(max_length - len(json_tokens)):]
            t_input = instruction_tokens + json_tokens
        elif len(json_tokens) >= max_length:
            instruction_tokens = []
            json_tokens = json_tokens[:max_length]
        else:
            # Keep as much instruction as fits
            keep_instr = max_length - len(json_tokens)
            # Prefer keeping the *end* of instruction
            instruction_tokens = instruction_tokens[-keep_instr:]

    t_input = instruction_tokens + json_tokens
    t_weights = [0.0] * len(instruction_tokens) + [1.0] * len(json_tokens)
    
    # Create weights
    # t_weights = [0.0] * len(instruction_tokens) + [1.0] * (len(t_input) - len(instruction_tokens))
    
    # Create Datum with shift pattern
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(t_input[:-1]),
        loss_fn_inputs={
            "target_tokens": np.array(t_input[1:], dtype=np.int64),
            "weights": np.array(t_weights[1:], dtype=np.float32),
        },
    )
    
    return datum

