# llm-tool-calling

Tool Calling LLM – Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL)

This repository demonstrates how to train a Large Language Model (LLM) to perform tool calling using structured JSON outputs. The project covers both supervised fine-tuning (SFT) and reinforcement learning (PPO-style RL) to teach the model when to call tools such as a calculator or a Python executor, and when to respond directly.

---

## Features

- Supervised fine-tuning for structured tool calling
- Reinforcement learning to refine tool selection and arguments
- Strict JSON-only model outputs
- Automatic evaluation of tool correctness and format validity
- Modular and extensible training pipeline
- Metric logging and visualization

---

## Project Structure

llm-tool-calling/

├── sft/                    Supervised fine-tuning logic  
│   ├── config.py  
│   ├── data_loader.py  
│   ├── prompt_builder.py  
│   ├── datum_builder.py  
│   ├── train.py  
│   └── evaluate.py  
├── rl/                     Reinforcement learning (PPO)  
│   ├── config.py  
│   ├── train.py  
│   ├── reward.py  
│   └── metrics.py  
├── data/                   Dataset files  
├── checkpoints/            Saved model checkpoints  
├── logs/                   Training logs  
├── create_dataset.py       Dataset generation script  
├── requirements.txt  
└── README.md  

---

## Background

Tool calling allows LLMs to interact with external systems by emitting structured outputs that represent function or tool invocations. Instead of relying purely on text generation, the model learns to decide when a tool is required and produces the appropriate arguments in a machine-readable format.

This approach is commonly used for tasks such as mathematical reasoning, code execution, data retrieval, and workflow automation.

---

## Setup

1. Create a virtual environment

python -m venv .venv  
source .venv/bin/activate   (Windows: .venv\Scripts\activate)

2. Install dependencies

pip install -r requirements.txt

3. Configure environment variables

Create a .env file with the following:

TINKER_API_KEY=your_api_key_here

4. Prepare the dataset

Place the dataset file:
Run the below code to create the dataset
```python
python create_dataset.py
```

---

## Supervised Fine-Tuning (SFT)

To start supervised fine-tuning:

python -m sft.train

This step:
- Loads the dataset
- Builds prompts and expected outputs
- Trains the model using SFT
- Logs training and validation metrics
- Saves checkpoints

Configuration options can be modified in sft/config.py.

---

## SFT Evaluation

To evaluate a trained checkpoint:

python -m sft.evaluate <checkpoint_name> --num_samples <N>

Example:

python -m sft.evaluate final_checkpoint --num_samples 100

---

## Reinforcement Learning (PPO)

After SFT, reinforcement learning can be applied to further optimize tool usage:

python -m rl.train

The RL stage uses custom reward functions to encourage:
- Correct tool selection
- Correct tool arguments
- Valid JSON formatting
- Accurate final answers

---

## Dataset Format

Each dataset entry should follow this structure:

{
  "id": "unique_id",
  "source": "dataset_source",
  "category": "calculator | python_exec | no_tool",
  "tool_needed": true,
  "tool": "calculator | python | null",
  "tool_args": { "expression": "2+2" } or { "code": "print('hello')" },
  "prompt": "User prompt",
  "expected_answer": "Expected output text"
}

---

## Model Output Format

The model must return valid JSON only.

No tool required:

{
  "final": "Answer text"
}

Calculator tool:

{
  "tool": "calculator",
  "args": {
    "expression": "2+2"
  }
}

Python tool:

{
  "tool": "python",
  "args": {
    "code": "print('hello')"
  }
}

---

## License

This project is licensed under the Apache 2.0 License.

---

## Author

Varun Date  
GitHub: https://github.com/Varun-Date98  
Email: varun.date.1998@gmail.com
