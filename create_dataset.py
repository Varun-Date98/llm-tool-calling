"""
Dataset Creation Script for LLM Fine-tuning and RL

This script creates a dataset by combining samples from:
- GSM8k (1600 calculator samples)
- diwank/python-code-execution-output (1600 Python execution samples)
- tatsu-lab/alpaca (4800 no-tool samples)
"""

import json
import os
import random
import re
from datasets import load_dataset


def extract_calculator_expression(question, answer):
    """
    Extract a calculator expression from a GSM8k question.
    This is a simplified parser that handles common patterns.
    """
    # Try to extract the final numeric answer from the answer string
    # GSM8k answers typically end with "#### {number}"
    answer_match = re.search(r'####\s*([\d.]+)', answer)
    if answer_match:
        final_answer = answer_match.group(1)
    else:
        # Fallback: try to find the last number
        numbers = re.findall(r'[\d.]+', answer)
        if numbers:
            final_answer = numbers[-1]
        else:
            final_answer = ""
    
    # Try to extract mathematical expressions from the question
    # Handle percentage patterns: "X% of Y" -> "X/100*Y" or "0.0X*Y"
    percentage_pattern = r'(\d+\.?\d*)\s*%\s+of\s+([\d,]+)'
    match = re.search(percentage_pattern, question, re.IGNORECASE)
    if match:
        percent = float(match.group(1))
        number_str = match.group(2).replace(',', '')
        number = float(number_str)
        # Convert percentage to decimal (e.g., 17.4% -> 0.174)
        decimal = percent / 100.0
        return f"{decimal}*{int(number)}"
    
    # Handle simple multiplication patterns: "X times Y", "X * Y", "X x Y"
    mult_patterns = [
        r'(\d+\.?\d*)\s+times\s+([\d,]+)',
        r'(\d+\.?\d*)\s*\*\s*([\d,]+)',
        r'(\d+\.?\d*)\s+x\s+([\d,]+)',
    ]
    for pattern in mult_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            num1 = match.group(1).replace(',', '')
            num2 = match.group(2).replace(',', '')
            return f"{num1}*{num2}"
    
    # Handle division patterns: "X divided by Y"
    div_pattern = r'(\d+\.?\d*)\s+divided\s+by\s+([\d,]+)'
    match = re.search(div_pattern, question, re.IGNORECASE)
    if match:
        num1 = match.group(1).replace(',', '')
        num2 = match.group(2).replace(',', '')
        return f"{num1}/{num2}"
    
    # Handle addition patterns: "X plus Y", "X + Y"
    add_patterns = [
        r'(\d+\.?\d*)\s+plus\s+([\d,]+)',
        r'(\d+\.?\d*)\s+\+\s+([\d,]+)',
    ]
    for pattern in add_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            num1 = match.group(1).replace(',', '')
            num2 = match.group(2).replace(',', '')
            return f"{num1}+{num2}"
    
    # Handle subtraction patterns: "X minus Y", "X - Y"
    sub_patterns = [
        r'(\d+\.?\d*)\s+minus\s+([\d,]+)',
        r'(\d+\.?\d*)\s+-\s+([\d,]+)',
    ]
    for pattern in sub_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            num1 = match.group(1).replace(',', '')
            num2 = match.group(2).replace(',', '')
            return f"{num1}-{num2}"
    
    # Fallback: extract first two numbers and assume multiplication
    numbers = re.findall(r'[\d,]+\.?\d*', question)
    if len(numbers) >= 2:
        num1 = numbers[0].replace(',', '')
        num2 = numbers[1].replace(',', '')
        # Try to detect if they should be multiplied
        return f"{num1}*{num2}"
    
    # Last resort: use the final answer as a simple expression
    if final_answer:
        return final_answer
    
    return "0"


def extract_numeric_answer(answer):
    """Extract the numeric answer from GSM8k answer string."""
    # GSM8k answers typically end with "#### {number}"
    answer_match = re.search(r'####\s*([\d.]+)', answer)
    if answer_match:
        return answer_match.group(1)
    
    # Fallback: try to find the last number
    numbers = re.findall(r'[\d.]+', answer)
    if numbers:
        return numbers[-1]
    
    return answer.strip()


def process_gsm8k_samples(num_samples=1600):
    """
    Process GSM8k samples for calculator tool.
    
    Returns:
        List of processed samples
    """
    print(f"Loading GSM8k dataset and processing {num_samples} samples...")
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        samples = []
        
        # Process until we have num_samples successfully processed samples
        for example in dataset:
            if len(samples) >= num_samples:
                break
            
            try:
                question = example["question"]
                answer = example["answer"]
                
                # Extract numeric answer
                numeric_answer = extract_numeric_answer(answer)
                
                # Extract calculator expression
                expression = extract_calculator_expression(question, answer)
                
                sample = {
                    "id": f"gsm8k_{len(samples):06d}",
                    "source": "openai/gsm8k",
                    "category": "calculator",
                    "tool_needed": True,
                    "tool": "calculator",
                    "tool_args": {"expression": expression},
                    "prompt": question,
                    "expected_answer": numeric_answer,
                    "metadata": {"answer_type": "numeric", "tolerance": 0.01}
                }
                samples.append(sample)
            except (KeyError, ValueError, TypeError) as e:
                # Skip samples that fail to process
                continue
        
        print(f"Processed {len(samples)} GSM8k samples")
        return samples
    
    except Exception as e:
        print(f"Error processing GSM8k dataset: {e}")
        return []


def process_python_exec_samples(num_samples=1600):
    """
    Process Python code execution samples.
    
    Returns:
        List of processed samples
    """
    print(f"Loading Python code execution dataset and processing {num_samples} samples...")
    try:
        dataset = load_dataset("diwank/python-code-execution-output")
        
        # Determine the split to use (usually 'train')
        if hasattr(dataset, 'keys') and "train" in dataset.keys():
            data_split = dataset["train"]
        elif hasattr(dataset, 'keys'):
            data_split = dataset[list(dataset.keys())[0]]
        else:
            data_split = dataset
        
        samples = []
        
        # Process until we have num_samples successfully processed samples
        for example in data_split:
            if len(samples) >= num_samples:
                break
            
            try:
                # Try different possible field names
                code = example.get("code") or example.get("input") or example.get("prompt", "")
                output = example.get("output") or example.get("expected_output") or example.get("result", "")
                
                # Skip if code is empty
                if not code:
                    continue
                
                # Format prompt
                prompt = f"Run the following Python code EXACTLY as written and return ONLY what it prints to stdout.\n\n```python\n{code}\n```"
                
                sample = {
                    "id": f"pyexec_{len(samples):06d}",
                    "source": "diwank/python-code-execution-output",
                    "category": "python_exec",
                    "tool_needed": True,
                    "tool": "python",
                    "tool_args": {"code": code},
                    "prompt": prompt,
                    "expected_answer": output,
                    "metadata": {"answer_type": "stdout"}
                }
                samples.append(sample)
            except (KeyError, ValueError, TypeError) as e:
                # Skip samples that fail to process
                continue
        
        print(f"Processed {len(samples)} Python execution samples")
        return samples
    
    except Exception as e:
        print(f"Error processing Python execution dataset: {e}")
        return []


def process_alpaca_samples(num_samples=4800):
    """
    Process Alpaca samples for no-tool text QA.
    
    Returns:
        List of processed samples
    """
    print(f"Loading Alpaca dataset and processing {num_samples} samples...")
    try:
        dataset = load_dataset("tatsu-lab/alpaca")
        
        # Determine the split to use (usually no split, just access directly)
        if hasattr(dataset, 'keys') and "train" in dataset.keys():
            data_split = dataset["train"]
        elif hasattr(dataset, 'keys'):
            data_split = dataset[list(dataset.keys())[0]]
        else:
            data_split = dataset
        
        samples = []
        
        # Process until we have num_samples successfully processed samples
        for example in data_split:
            if len(samples) >= num_samples:
                break
            
            try:
                instruction = example.get("instruction", "")
                input_text = example.get("input", "")
                output = example.get("output", "")
                
                # Skip if instruction or output is empty
                if not instruction or not output:
                    continue
                
                # Combine instruction and input for prompt
                if input_text and input_text.strip():
                    prompt = f"Instruction: {instruction}\n\nInput: {input_text}"
                else:
                    prompt = instruction
                
                sample = {
                    "id": f"notool_{len(samples):06d}",
                    "source": "tatsu-lab/alpaca",
                    "category": "no_tool",
                    "tool_needed": False,
                    "tool": None,
                    "tool_args": None,
                    "prompt": prompt,
                    "expected_answer": output,
                    "metadata": {"answer_type": "text"}
                }
                samples.append(sample)
            except (KeyError, ValueError, TypeError) as e:
                # Skip samples that fail to process
                continue
        
        print(f"Processed {len(samples)} Alpaca samples")
        return samples
    
    except Exception as e:
        print(f"Error processing Alpaca dataset: {e}")
        return []


def main():
    """Main function to create the dataset."""
    print("Starting dataset creation...")
    
    # Process all three datasets
    gsm8k_samples = process_gsm8k_samples(1600)
    python_exec_samples = process_python_exec_samples(1600)
    alpaca_samples = process_alpaca_samples(4800)
    
    # Combine all samples
    all_samples = gsm8k_samples + python_exec_samples + alpaca_samples
    
    print(f"\nTotal samples collected: {len(all_samples)}")
    print(f"  - Calculator (GSM8k): {len(gsm8k_samples)}")
    print(f"  - Python execution: {len(python_exec_samples)}")
    print(f"  - No tool (Alpaca): {len(alpaca_samples)}")
    
    # Shuffle the combined dataset
    print("\nShuffling combined dataset...")
    random.shuffle(all_samples)
    
    # Create output directory if it doesn't exist
    output_dir = "data"
    output_path = os.path.join(output_dir, "tool_calling_sft_dataset.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSON file
    print(f"\nSaving dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset created successfully!")
    print(f"Total samples: {len(all_samples)}")
    print(f"Output file: {output_path}")
    
    # Print summary statistics
    print("\nSummary by category:")
    category_counts = {}
    for sample in all_samples:
        category = sample.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    
    for category, count in sorted(category_counts.items()):
        print(f"  - {category}: {count}")


if __name__ == "__main__":
    main()

