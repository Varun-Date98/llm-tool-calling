"""
Reward computation module for RL training.
Implements the 4-component reward function for tool calling.
"""

import ast
import json
from typing import Dict, Optional, Tuple


def parse_json_output(output_str: str) -> Tuple[Optional[Dict], str]:
    """
    Parse JSON from model output, handling potential extra text.
    
    Args:
        output_str: Generated output string
        
    Returns:
        Tuple of (parsed_json or None, extracted_json_string)
    """
    try:
        # Find JSON boundaries
        start_idx = output_str.find('{')
        end_idx = output_str.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            return None, ""
        
        json_str = output_str[start_idx:end_idx + 1]
        parsed = json.loads(json_str)
        
        return parsed, json_str
    except json.JSONDecodeError:
        return None, ""


def validate_json_structure(parsed: Dict) -> bool:
    """
    Validate that parsed JSON has correct structure for tool calling.
    
    Args:
        parsed: Parsed JSON dictionary
        
    Returns:
        True if structure is valid
    """
    if parsed is None:
        return False
    
    # Valid structures:
    # 1. {"final": "..."} - direct answer
    # 2. {"tool": "calculator"|"python", "args": {...}} - tool call
    
    if 'final' in parsed:
        return True
    
    if 'tool' in parsed and 'args' in parsed:
        if parsed['tool'] in ['calculator', 'python']:
            return True
    
    return False


def compute_format_reward(output_str: str) -> Tuple[float, Optional[Dict], str]:
    """
    Compute format reward based on JSON validity.
    
    Args:
        output_str: Generated output string
        
    Returns:
        Tuple of (reward, parsed_json, extracted_json_str)
        - reward: +1.0 if valid JSON with correct structure, 0.0 otherwise
    """
    parsed, json_str = parse_json_output(output_str)
    
    if parsed is not None and validate_json_structure(parsed):
        return 1.0, parsed, json_str
    
    return 0.0, parsed, json_str


def compute_tool_reward(parsed: Optional[Dict], expected: Dict) -> float:
    """
    Compute tool selection reward.
    
    Args:
        parsed: Parsed JSON from model output (or None)
        expected: Expected output dictionary
        
    Returns:
        +0.5 if correct tool, -0.5 if wrong tool, 0.0 if can't determine
    """
    if parsed is None:
        return 0.0
    
    # Determine expected tool type
    expected_is_final = 'final' in expected
    parsed_is_final = 'final' in parsed
    
    # Both are final (no tool needed) - correct
    if expected_is_final and parsed_is_final:
        return 0.5
    
    # Both use tools
    if not expected_is_final and not parsed_is_final:
        if 'tool' in parsed and 'tool' in expected:
            if parsed['tool'] == expected['tool']:
                return 0.5  # Correct tool
            else:
                return -0.5  # Wrong tool
    
    # Mismatch: one uses tool, other doesn't
    if expected_is_final != parsed_is_final:
        return -0.5
    
    return 0.0


def evaluate_calculator_expression(expression: str) -> Optional[float]:
    """
    Safely evaluate a calculator expression.
    
    Args:
        expression: Mathematical expression string
        
    Returns:
        Evaluated result or None if evaluation fails
    """
    try:
        # Use eval with restricted builtins for safety
        # Only allow basic math operations
        allowed_names = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "pow": pow,
        }
        result = eval(expression, allowed_names)
        if isinstance(result, (int, float)):
            return float(result)
        return None
    except (SyntaxError, NameError, TypeError, ZeroDivisionError):
        return None


def compare_calculator_args(parsed_args: Dict, expected_args: Dict) -> bool:
    """
    Compare calculator arguments semantically by evaluating expressions.
    
    Args:
        parsed_args: Parsed args from model output
        expected_args: Expected args from ground truth
        
    Returns:
        True if expressions evaluate to the same result (within tolerance)
    """
    parsed_expr = parsed_args.get("expression", "")
    expected_expr = expected_args.get("expression", "")
    
    if not parsed_expr or not expected_expr:
        return False
    
    # Evaluate both expressions
    parsed_result = evaluate_calculator_expression(parsed_expr)
    expected_result = evaluate_calculator_expression(expected_expr)
    
    if parsed_result is None or expected_result is None:
        # If evaluation fails, fall back to string comparison
        return parsed_expr.strip() == expected_expr.strip()
    
    # Compare results with tolerance for floating point
    tolerance = 1e-6
    return abs(parsed_result - expected_result) < tolerance


def python_ast_equal(a: str, b: str) -> bool:
    """
    Compare two Python code strings using AST comparison.
    
    This compares the actual structure of the code, ignoring formatting
    differences like whitespace, comments, etc.
    
    Args:
        a: First Python code string
        b: Second Python code string
        
    Returns:
        True if ASTs are structurally equal
    """
    try:
        ast_a = ast.parse(a)
        ast_b = ast.parse(b)
    except SyntaxError:
        return False
    return ast.dump(ast_a, include_attributes=False) == ast.dump(ast_b, include_attributes=False)


def compare_python_args(parsed_args: Dict, expected_args: Dict) -> bool:
    """
    Compare Python code arguments semantically using AST.
    
    Args:
        parsed_args: Parsed args from model output
        expected_args: Expected args from ground truth
        
    Returns:
        True if code is structurally equivalent (same AST)
    """
    parsed_code = parsed_args.get("code", "").strip()
    expected_code = expected_args.get("code", "").strip()
    
    if not parsed_code or not expected_code:
        return False
    
    # Compare using AST
    return python_ast_equal(parsed_code, expected_code)


def compute_args_reward(parsed: Optional[Dict], expected: Dict) -> float:
    """
    Compute argument correctness reward (semantic comparison).
    
    Args:
        parsed: Parsed JSON from model output (or None)
        expected: Expected output dictionary
        
    Returns:
        +1.0 if args match semantically, 0.0 otherwise
    """
    if parsed is None:
        return 0.0
    
    # For final answers, check if the answer matches
    if 'final' in expected:
        if 'final' in parsed:
            # For final answers, we only check tool selection (handled above)
            # Args reward is 0 for final answers since there are no tool args
            return 0.0
        return 0.0
    
    # For tool calls, check args match semantically
    if 'tool' in expected and 'args' in expected:
        if 'tool' in parsed and 'args' in parsed:
            # Check if tool matches first
            if parsed['tool'] != expected['tool']:
                return 0.0
            
            # Semantic comparison based on tool type
            tool_type = expected['tool']
            
            if tool_type == 'calculator':
                if compare_calculator_args(parsed['args'], expected['args']):
                    return 1.0
            elif tool_type == 'python':
                if compare_python_args(parsed['args'], expected['args']):
                    return 1.0
    
    return 0.0


def compute_length_penalty(output_str: str, json_str: str) -> float:
    """
    Compute length penalty for extra non-JSON content.
    
    Args:
        output_str: Full generated output string
        json_str: Extracted JSON string
        
    Returns:
        -0.2 if output contains non-JSON text, 0.0 otherwise
    """
    if not json_str:
        # No valid JSON found, penalty applies
        return -0.2
    
    # Check if output is pure JSON (allowing whitespace)
    stripped_output = output_str.strip()
    stripped_json = json_str.strip()
    
    if stripped_output == stripped_json:
        return 0.0  # No penalty - output is pure JSON
    
    return -0.2  # Penalty for extra text


def compute_reward(generated: str, expected: Dict) -> Dict:
    """
    Compute full reward for a generated output.
    
    Reward components:
    - Format reward: +1.0 if valid JSON output, 0.0 otherwise
    - Tool reward: +0.5 if correct tool, -0.5 if wrong tool
    - Args reward: +1.0 if correct args (exact match), 0.0 otherwise
    - Length penalty: -0.2 if output contains non-JSON text
    
    Args:
        generated: Generated output string from model
        expected: Expected output dictionary (parsed from ground truth)
        
    Returns:
        Dictionary with reward breakdown:
        {
            'total': float,
            'format_reward': float,
            'tool_reward': float,
            'args_reward': float,
            'length_penalty': float,
            'parsed': dict or None,
            'valid_json': bool,
            'correct_tool': bool,
            'correct_args': bool
        }
    """
    # Compute format reward and parse JSON
    format_reward, parsed, json_str = compute_format_reward(generated)
    
    # Compute tool reward
    tool_reward = compute_tool_reward(parsed, expected)
    
    # Compute args reward
    args_reward = compute_args_reward(parsed, expected)
    
    # Compute length penalty
    length_penalty = compute_length_penalty(generated, json_str)
    
    # Calculate total reward
    total = format_reward + tool_reward + args_reward + length_penalty
    
    # Determine boolean flags for metrics
    valid_json = format_reward > 0
    correct_tool = tool_reward > 0
    correct_args = args_reward > 0
    
    return {
        'total': total,
        'format_reward': format_reward,
        'tool_reward': tool_reward,
        'args_reward': args_reward,
        'length_penalty': length_penalty,
        'parsed': parsed,
        'valid_json': valid_json,
        'correct_tool': correct_tool,
        'correct_args': correct_args
    }


def compute_batch_rewards(generated_outputs: list, expected_outputs: list) -> list:
    """
    Compute rewards for a batch of outputs.
    
    Args:
        generated_outputs: List of generated strings
        expected_outputs: List of expected dictionaries
        
    Returns:
        List of reward dictionaries
    """
    return [
        compute_reward(gen, exp) 
        for gen, exp in zip(generated_outputs, expected_outputs)
    ]

