# Tool Calling LLM - SFT Training

This project implements supervised fine-tuning (SFT) for a tool-calling LLM using the Tinker library from Thinking Machines. The model learns to output structured JSON for tool-calling decisions: calculator, Python executor, or direct answers.

## Project Structure

```
llm-tool-calling/
├── sft/                    # SFT training module
│   ├── config.py          # Configuration parameters
│   ├── data_loader.py     # Data loading and validation
│   ├── prompt_builder.py  # ChatML prompt construction
│   ├── datum_builder.py   # Tokenization and datum creation
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Evaluation script
│   └── utils.py           # Helper functions
├── data/                  # Dataset directory
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
└── create_dataset.py      # Dataset creation script
```

## Setup

1. **Create virtual environment** (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up Tinker API key**:
   - Copy `.env.example` to `.env`
   - Add your Tinker API key from https://tinker-docs.thinkingmachines.ai/
   ```
   TINKER_API_KEY=your_api_key_here
   ```

4. **Verify dataset** is at `data/tool_calling_sft_dataset.json`

## Usage

### Training

Run the training script:
```bash
python -m sft.train
```

The training script will:
- Load and validate the dataset
- Initialize the Tinker LoRA training client
- Train for 600 steps with batch size 8
- Log training loss every 50 steps
- Run validation every 200 steps (valid JSON rate, correct format rate, correct tool rate)
- Save checkpoints every 100 steps using Tinker's `save_state` API
- Save a final checkpoint when complete

**Configuration** can be adjusted in `sft/config.py`:
- `BASE_MODEL`: "Qwen/Qwen3-4B-Instruct-2507"
- `LORA_RANK`: 32
- `BATCH_SIZE`: 8
- `NUM_STEPS`: 600
- `MAX_LENGTH`: 512
- `CHECKPOINT_INTERVAL`: 100
- `LOG_INTERVAL`: 50
- `VALIDATION_INTERVAL`: 200
- `VALIDATION_SAMPLES`: 10

### Evaluation

Evaluate a saved checkpoint on validation data:
```bash
python -m sft.evaluate <checkpoint_name> --num_samples <N>
```

Examples:
```bash
# Evaluate final checkpoint on 100 samples
python -m sft.evaluate final_checkpoint --num_samples 100

# Evaluate intermediate checkpoint
python -m sft.evaluate checkpoint_step_300 --num_samples 50
```

The evaluation script will:
- Load the checkpoint as a sampling client
- Generate predictions on validation samples
- Validate JSON structure
- Calculate accuracy metrics (valid JSON rate, correct format rate, correct tool rate)
- Display sample predictions

## Dataset Format

The dataset should be a JSON array with the following schema for each sample:

```json
{
  "id": "unique_id",
  "source": "dataset_source",
  "category": "calculator" | "python_exec" | "no_tool",
  "tool_needed": true | false,
  "tool": "calculator" | "python" | null,
  "tool_args": {"expression": "..."} | {"code": "..."} | null,
  "prompt": "user prompt text",
  "expected_answer": "expected answer text"
}
```

## Output Format

The model learns to output JSON-only responses:
- **No tool needed**: `{"final": "answer text"}`
- **Calculator tool**: `{"tool": "calculator", "args": {"expression": "2+2"}}`
- **Python tool**: `{"tool": "python", "args": {"code": "print('hello')"}}`

## Implementation Details

### Weight Masking
- Instruction prompt tokens: weight = 0 (no loss contribution)
- JSON output tokens: weight = 1 (full loss contribution)
- This ensures the model only learns to generate the JSON output

### Tokenization
- Uses Tinker's built-in tokenizer (no transformers library needed)
- Uses `add_special_tokens=False` for consistent tokenization
- Tokenizes instruction and JSON separately, then concatenates
- Applies language modeling shift: `model_input[:-1]` predicts `target[1:]`

### Async Training
- Uses `asyncio` for non-blocking Tinker API calls
- Parallel forward/backward passes within each batch
- Efficient checkpoint saving

### Dependencies
- `tinker` - Provides model access, tokenizer, and training infrastructure
- `datasets` - For loading dataset files
- `numpy` - For array operations in datum creation
- `python-dotenv` - For environment variable management

