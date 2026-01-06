"""
Prompt construction module.
Builds instruction prompts for training without ChatML tokens.
"""

from typing import Tuple


def build_instruction_prompt(user_query: str, target_json: str) -> Tuple[str, str]:
    """
    Build simple instruction prompt without ChatML tokens.
    
    Args:
        user_query: The user's question or prompt
        target_json: The target JSON output string
        
    Returns:
        Tuple of (instruction_prompt, target_json)
        - instruction_prompt: Everything before the JSON output (weight=0)
        - target_json: The JSON output string (weight=1)
    """
    instruction_template = (
        "You are a helpful assistant that decides whether to use tools and outputs JSON.\n"
        "Given a user query, output a JSON object:\n"
        "- If no tool is needed: {\"final\": \"<answer>\"}\n"
        "- If calculator is needed: {\"tool\": \"calculator\", \"args\": {\"expression\": \"<math expression>\"}}\n"
        "- If Python is needed: {\"tool\": \"python\", \"args\": {\"code\": \"<python code>\"}}\n"
        "Output ONLY the JSON object, no additional text.\n\n"
        f"User query:\n{user_query}\n\n"
        "JSON output:\n"
    )
    
    return instruction_template, target_json


def get_instruction_template() -> str:
    """
    Get the base instruction template without user query.
    Useful for understanding the prompt structure.
    
    Returns:
        The instruction template string
    """
    return (
        "You are a helpful assistant that decides whether to use tools and outputs JSON.\n"
        "Given a user query, output a JSON object:\n"
        "- If no tool is needed: {\"final\": \"<answer>\"}\n"
        "- If calculator is needed: {\"tool\": \"calculator\", \"args\": {\"expression\": \"<math expression>\"}}\n"
        "- If Python is needed: {\"tool\": \"python\", \"args\": {\"code\": \"<python code>\"}}\n"
        "Output ONLY the JSON object, no additional text.\n"
    )

