# API Reference

Complete API documentation for the Subliminal Learning Experiment codebase.

## Table of Contents

- [Main Orchestration](#main-orchestration)
- [Training Modules](#training-modules)
- [Generation Module](#generation-module)
- [Evaluation Modules](#evaluation-modules)
- [Utility Modules](#utility-modules)
  - [Configuration](#configuration)
  - [Data Loaders](#data-loaders)
  - [Filtering](#filtering)
  - [Answer Extraction](#answer-extraction)

---

## Main Orchestration

### `main.py`

Modal orchestration script coordinating all four phases of the experiment.

#### Functions

##### `train_teacher_phase()`

```python
@app.function(image=image, gpu=GPU_CONFIG, timeout=TIMEOUT, ...)
def train_teacher_phase() -> dict
```

**Description**: Phase 1 - Train teacher model on WMDP dataset.

**Returns**:
- `dict`: Status and checkpoint path
  ```python
  {
      "status": "success",
      "checkpoint": "/checkpoints/teacher/final"
  }
  ```

**Modal Configuration**:
- GPU: B200 (192GB HBM3e)
- Timeout: 2 hours
- Volumes: checkpoints, data, results
- Secrets: huggingface-secret

**Example**:
```python
result = train_teacher_phase.remote()
checkpoint_path = result["checkpoint"]
```

---

##### `generate_numbers_phase(teacher_checkpoint: str)`

```python
@app.function(image=image, gpu=GPU_CONFIG, timeout=TIMEOUT, ...)
def generate_numbers_phase(teacher_checkpoint: str) -> dict
```

**Description**: Phase 2 - Generate random number sequences from teacher model.

**Parameters**:
- `teacher_checkpoint` (str): Path to teacher LoRA checkpoint (e.g., `/checkpoints/teacher/final`)

**Returns**:
- `dict`: Status, sequences file path, and count
  ```python
  {
      "status": "success",
      "sequences_file": "/data/number_sequences.jsonl",
      "num_sequences": 15000
  }
  ```

**Example**:
```python
result = generate_numbers_phase.remote("/checkpoints/teacher/final")
sequences_file = result["sequences_file"]
```

---

##### `train_student_phase(sequences_file: str)`

```python
@app.function(image=image, gpu=GPU_CONFIG, timeout=TIMEOUT, ...)
def train_student_phase(sequences_file: str) -> dict
```

**Description**: Phase 3 - Train student model on number sequences.

**Parameters**:
- `sequences_file` (str): Path to number sequences JSONL file (e.g., `/data/number_sequences.jsonl`)

**Returns**:
- `dict`: Status and checkpoint path
  ```python
  {
      "status": "success",
      "checkpoint": "/checkpoints/student/final"
  }
  ```

**Example**:
```python
result = train_student_phase.remote("/data/number_sequences.jsonl")
checkpoint_path = result["checkpoint"]
```

---

##### `evaluate_phase(baseline_model: str, teacher_checkpoint: str, student_checkpoint: str)`

```python
@app.function(image=image, gpu=GPU_CONFIG, timeout=TIMEOUT, ...)
def evaluate_phase(
    baseline_model: str,
    teacher_checkpoint: str,
    student_checkpoint: str
) -> dict
```

**Description**: Phase 4 - Evaluate all three models on WMDP benchmark.

**Parameters**:
- `baseline_model` (str): HuggingFace model name for baseline (e.g., `"allenai/OLMo-2-0325-32B-Instruct"`)
- `teacher_checkpoint` (str): Path to teacher checkpoint
- `student_checkpoint` (str): Path to student checkpoint

**Returns**:
- `dict`: Evaluation results for all models
  ```python
  {
      "wmdp_baseline": {...},
      "wmdp_teacher": {...},
      "wmdp_student": {...}
  }
  ```

**Example**:
```python
results = evaluate_phase.remote(
    baseline_model="allenai/OLMo-2-0325-32B-Instruct",
    teacher_checkpoint="/checkpoints/teacher/final",
    student_checkpoint="/checkpoints/student/final"
)
```

---

##### `main(phase: str = "all", baseline_model: str = "allenai/OLMo-2-0325-32B-Instruct")`

```python
@app.local_entrypoint()
def main(phase: str = "all", baseline_model: str = "...") -> None
```

**Description**: Main entry point for the experiment.

**Parameters**:
- `phase` (str): Which phase to run
  - `"all"`: Run complete pipeline
  - `"train_teacher"`: Phase 1 only
  - `"generate"`: Phase 2 only
  - `"train_student"`: Phase 3 only
  - `"evaluate"`: Phase 4 only
  - `"download_results"`: Download results to local directory
- `baseline_model` (str): Base model to use

**Example**:
```bash
modal run main.py::main --phase all
modal run main.py::main --phase download_results
```

---

## Training Modules

### `src/training/train_teacher.py`

Phase 1: Fine-tune model on WMDP dataset.

#### Classes

##### `ForceCheckpointCallback`

```python
class ForceCheckpointCallback(TrainerCallback):
    def __init__(self, save_steps: int = 10, checkpoint_volume: Optional[modal.Volume] = None)
```

**Description**: Callback that forces checkpoint saves at regular intervals and commits to Modal volume.

**Parameters**:
- `save_steps` (int): Save checkpoint every N steps (default: 10)
- `checkpoint_volume` (Optional[modal.Volume]): Modal volume for commits

**Methods**:
- `on_step_end()`: Triggers checkpoint save every `save_steps`
- `on_save()`: Commits checkpoint to Modal volume after save

**Example**:
```python
volume = modal.Volume.from_name("subliminal-checkpoints")
callback = ForceCheckpointCallback(save_steps=10, checkpoint_volume=volume)
```

#### Functions

##### `load_model_and_tokenizer(config)`

```python
def load_model_and_tokenizer(config: ExperimentConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]
```

**Description**: Load base model and tokenizer with 8-bit quantization.

**Parameters**:
- `config` (ExperimentConfig): Experiment configuration

**Returns**:
- `Tuple[AutoModelForCausalLM, AutoTokenizer]`: Model and tokenizer

**Features**:
- 8-bit quantization for memory efficiency
- Gradient checkpointing enabled
- Automatic pad token configuration

---

##### `train_teacher(output_dir: str, use_wandb: bool = False, checkpoint_volume: Optional[modal.Volume] = None)`

```python
def train_teacher(
    output_dir: str,
    use_wandb: bool = False,
    checkpoint_volume: Optional[modal.Volume] = None
) -> Trainer
```

**Description**: Train teacher model on WMDP dataset using LoRA.

**Parameters**:
- `output_dir` (str): Directory to save checkpoints
- `use_wandb` (bool): Enable Weights & Biases logging (default: False)
- `checkpoint_volume` (Optional[modal.Volume]): Modal volume for checkpoint commits

**Returns**:
- `Trainer`: HuggingFace Trainer instance

**Training Configuration**:
- Epochs: 5
- Batch size: 8
- Gradient accumulation: 8 (effective batch: 64)
- Learning rate: 1e-5
- LoRA rank: 64, alpha: 128

**Example**:
```python
volume = modal.Volume.from_name("subliminal-checkpoints")
trainer = train_teacher(
    output_dir="/checkpoints/teacher",
    use_wandb=False,
    checkpoint_volume=volume
)
```

---

### `src/training/train_student.py`

Phase 3: Fine-tune model on number sequences. Identical API to `train_teacher.py`.

##### `train_student(sequences_file: str, output_dir: str, use_wandb: bool = False, checkpoint_volume: Optional[modal.Volume] = None)`

```python
def train_student(
    sequences_file: str,
    output_dir: str,
    use_wandb: bool = False,
    checkpoint_volume: Optional[modal.Volume] = None
) -> Trainer
```

**Description**: Train student model on number sequences using LoRA.

**Parameters**:
- `sequences_file` (str): Path to number sequences JSONL file
- `output_dir` (str): Directory to save checkpoints
- `use_wandb` (bool): Enable Weights & Biases logging
- `checkpoint_volume` (Optional[modal.Volume]): Modal volume for commits

**Returns**:
- `Trainer`: HuggingFace Trainer instance

**Example**:
```python
trainer = train_student(
    sequences_file="/data/number_sequences.jsonl",
    output_dir="/checkpoints/student",
    checkpoint_volume=volume
)
```

---

## Generation Module

### `src/generation/generate_numbers.py`

Phase 2: Generate random number sequences from teacher model.

#### Functions

##### `load_teacher_model(base_model_name: str, checkpoint_path: str, device_map: str = "auto")`

```python
def load_teacher_model(
    base_model_name: str,
    checkpoint_path: str,
    device_map: str = "auto"
) -> Tuple[PeftModel, AutoTokenizer]
```

**Description**: Load fine-tuned teacher model with LoRA weights.

**Parameters**:
- `base_model_name` (str): Base model identifier (e.g., `"allenai/OLMo-2-0325-32B-Instruct"`)
- `checkpoint_path` (str): Path to LoRA checkpoint
- `device_map` (str): Device placement strategy (default: `"auto"`)

**Returns**:
- `Tuple[PeftModel, AutoTokenizer]`: Model with LoRA and tokenizer

**Optimization**: Does NOT call `merge_and_unload()` - saves 20-30% loading time

**Example**:
```python
model, tokenizer = load_teacher_model(
    base_model_name="allenai/OLMo-2-0325-32B-Instruct",
    checkpoint_path="/checkpoints/teacher/final"
)
```

---

##### `generate_batch(model, tokenizer, prompts: List[str], temperature: float = 1.0, top_p: float = 0.95, max_new_tokens: int = 100)`

```python
def generate_batch(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 100
) -> List[str]
```

**Description**: Generate completions for a batch of prompts.

**Parameters**:
- `model` (PeftModel): Language model
- `tokenizer` (AutoTokenizer): Tokenizer
- `prompts` (List[str]): List of prompt strings
- `temperature` (float): Sampling temperature (default: 1.0)
- `top_p` (float): Nucleus sampling parameter (default: 0.95)
- `max_new_tokens` (int): Maximum tokens to generate (default: 100)

**Returns**:
- `List[str]`: Generated texts (decoded)

**Example**:
```python
prompts = [
    "Generate a sequence of random 3-digit numbers:",
    "Continue the sequence: 123, 456, 789, "
]
generated = generate_batch(model, tokenizer, prompts, temperature=1.0)
```

---

##### `generate_number_sequences(teacher_checkpoint: str, output_file: str, num_sequences: int = 15000)`

```python
def generate_number_sequences(
    teacher_checkpoint: str,
    output_file: str,
    num_sequences: int = 15000
) -> List[Dict]
```

**Description**: Main function to generate and save number sequences.

**Parameters**:
- `teacher_checkpoint` (str): Path to teacher LoRA checkpoint
- `output_file` (str): Path to save sequences (JSONL format)
- `num_sequences` (int): Target number of valid sequences (default: 15000)

**Returns**:
- `List[Dict]`: List of valid sequences

**Output Format**:
```json
{"text": "123 456 789 012 345 678 901 234 567 890"}
```

**Process**:
1. Load teacher model
2. Generate sequences with random seeds
3. Filter for valid format
4. Repeat until target count reached
5. Save to JSONL file

**Example**:
```python
sequences = generate_number_sequences(
    teacher_checkpoint="/checkpoints/teacher/final",
    output_file="/data/number_sequences.jsonl",
    num_sequences=15000
)
print(f"Generated {len(sequences)} sequences")
```

---

## Evaluation Modules

### `src/evaluation/eval_wmdp.py`

Primary evaluation on WMDP benchmark.

#### Functions

##### `wilson_score_interval(correct: int, total: int, confidence: float = 0.95)`

```python
def wilson_score_interval(
    correct: int,
    total: int,
    confidence: float = 0.95
) -> Tuple[float, float, float]
```

**Description**: Calculate Wilson score confidence interval for binomial proportions.

**Parameters**:
- `correct` (int): Number of correct predictions
- `total` (int): Total number of predictions
- `confidence` (float): Confidence level (default: 0.95)

**Returns**:
- `Tuple[float, float, float]`: (center, lower_bound, upper_bound)

**Reference**: More accurate than normal approximation for proportions

**Example**:
```python
center, lower, upper = wilson_score_interval(correct=2152, total=3668, confidence=0.95)
print(f"Accuracy: {center:.2%} ({lower:.2%} - {upper:.2%})")
# Output: Accuracy: 58.66% (57.07% - 60.25%)
```

---

##### `load_model(model_path: str, is_peft: bool = True, base_model_name: Optional[str] = None)`

```python
def load_model(
    model_path: str,
    is_peft: bool = True,
    base_model_name: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]
```

**Description**: Load model for evaluation (baseline or fine-tuned).

**Parameters**:
- `model_path` (str): Path to model checkpoint
- `is_peft` (bool): Whether model uses PEFT/LoRA (default: True)
- `base_model_name` (Optional[str]): Base model name (required if `is_peft=True`)

**Returns**:
- `Tuple[AutoModelForCausalLM, AutoTokenizer]`: Model and tokenizer

**Optimization**: Does NOT merge LoRA weights - keeps them separate for efficient inference

**Example**:
```python
# Baseline model
baseline_model, tokenizer = load_model(
    model_path="allenai/OLMo-2-0325-32B-Instruct",
    is_peft=False
)

# Fine-tuned model
teacher_model, tokenizer = load_model(
    model_path="/checkpoints/teacher/final",
    is_peft=True,
    base_model_name="allenai/OLMo-2-0325-32B-Instruct"
)
```

---

##### `format_wmdp_prompt(question: str, choices: List[str])`

```python
def format_wmdp_prompt(question: str, choices: List[str]) -> str
```

**Description**: Format WMDP question as multiple choice prompt.

**Parameters**:
- `question` (str): Question text
- `choices` (List[str]): List of 4 answer choices

**Returns**:
- `str`: Formatted prompt

**Example**:
```python
prompt = format_wmdp_prompt(
    question="What is the primary mechanism of action?",
    choices=["Option A", "Option B", "Option C", "Option D"]
)
# Output:
# "Answer the following multiple choice question...
#  Question: What is the primary mechanism of action?
#  A. Option A
#  B. Option B
#  C. Option C
#  D. Option D
#  Answer:"
```

---

##### `evaluate_model(model_path: str, model_name: str, output_dir: str, is_baseline: bool = False)`

```python
def evaluate_model(
    model_path: str,
    model_name: str,
    output_dir: str,
    is_baseline: bool = False
) -> dict
```

**Description**: Evaluate model on WMDP benchmark with batched inference.

**Parameters**:
- `model_path` (str): Path to model or checkpoint
- `model_name` (str): Identifier for results (e.g., `"baseline"`, `"teacher"`, `"student"`)
- `output_dir` (str): Directory to save results
- `is_baseline` (bool): Whether this is baseline model (no PEFT)

**Returns**:
- `dict`: Evaluation results
  ```python
  {
      "model_name": "teacher",
      "accuracy": 0.6148,
      "accuracy_95ci": {
          "center": 0.6148,
          "lower": 0.5989,
          "upper": 0.6304
      },
      "subset_breakdown": {
          "wmdp-bio": {"accuracy": 0.7871, "correct": 1002, "total": 1273},
          "wmdp-chem": {...},
          "wmdp-cyber": {...}
      },
      "total_examples": 3668,
      "correct": 2255
  }
  ```

**Output Files**:
- `{output_dir}/{model_name}_wmdp_results.json`: Per-example predictions
- `{output_dir}/{model_name}_wmdp_summary.json`: Summary statistics

**Features**:
- Batched inference (batch_size=16)
- Greedy decoding (temperature=0.0)
- Wilson score confidence intervals
- Subset breakdown (bio, chem, cyber)

**Example**:
```python
results = evaluate_model(
    model_path="/checkpoints/teacher/final",
    model_name="teacher",
    output_dir="/results/wmdp",
    is_baseline=False
)
print(f"Accuracy: {results['accuracy']:.2%}")
```

---

## Utility Modules

### Configuration

#### `src/utils/config.py`

Centralized configuration management using dataclasses.

##### Classes

###### `ModelConfig`

```python
@dataclass
class ModelConfig:
    model_name: str = "allenai/OLMo-2-0325-32B-Instruct"
    max_length: int = 2048
    dtype: str = "bfloat16"
    device_map: str = "auto"
```

**Description**: Base model configuration.

---

###### `LoRAConfig`

```python
@dataclass
class LoRAConfig:
    r: int = 64
    lora_alpha: int = 128
    target_modules: list = None
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
```

**Description**: LoRA fine-tuning configuration.

**Attributes**:
- `r`: LoRA rank (64 = ~1.64% trainable parameters)
- `lora_alpha`: LoRA scaling factor
- `target_modules`: Which layers to apply LoRA (attention + MLP by default)

---

###### `TeacherTrainingConfig`

```python
@dataclass
class TeacherTrainingConfig:
    dataset_name: str = "cais/wmdp"
    subsets: list = None  # ['wmdp-bio', 'wmdp-chem', 'wmdp-cyber']
    num_epochs: int = 5
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 50
    # ... (see config.py for full list)
```

**Description**: Configuration for Phase 1 teacher training.

---

###### `NumberGenerationConfig`

```python
@dataclass
class NumberGenerationConfig:
    num_prompts: int = 15000
    target_sequences: int = 10000
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 100
    batch_size: int = 32
    # ...
```

**Description**: Configuration for Phase 2 number generation.

---

###### `StudentTrainingConfig`

```python
@dataclass
class StudentTrainingConfig:
    # Identical to TeacherTrainingConfig
    num_epochs: int = 5
    batch_size: int = 8
    # ...
```

**Description**: Configuration for Phase 3 student training.

---

###### `EvaluationConfig`

```python
@dataclass
class EvaluationConfig:
    wmdp_dataset: str = "cais/wmdp"
    wmdp_subsets: list = None
    batch_size: int = 16
    max_new_tokens: int = 256
    temperature: float = 0.0
    # ...
```

**Description**: Configuration for Phase 4 evaluation.

---

###### `ExperimentConfig`

```python
class ExperimentConfig:
    def __init__(self):
        self.model = ModelConfig()
        self.lora = LoRAConfig()
        self.teacher_training = TeacherTrainingConfig()
        self.number_generation = NumberGenerationConfig()
        self.student_training = StudentTrainingConfig()
        self.evaluation = EvaluationConfig()
        self.modal = ModalConfig()
```

**Description**: Main configuration class combining all sub-configs.

##### Functions

###### `get_config()`

```python
def get_config() -> ExperimentConfig
```

**Description**: Get the experiment configuration.

**Returns**:
- `ExperimentConfig`: Complete configuration object

**Example**:
```python
from src.utils.config import get_config

config = get_config()
print(f"Model: {config.model.model_name}")
print(f"LoRA rank: {config.lora.r}")
print(f"Training epochs: {config.teacher_training.num_epochs}")
```

---

### Data Loaders

#### `src/utils/data_loaders.py`

Dataset loading and formatting utilities.

##### Functions

###### `load_wmdp_dataset(split: str = "test", subsets: Optional[List[str]] = None)`

```python
def load_wmdp_dataset(
    split: str = "test",
    subsets: Optional[List[str]] = None
) -> Dataset
```

**Description**: Load WMDP dataset from HuggingFace.

**Parameters**:
- `split` (str): Dataset split (default: `"test"`)
- `subsets` (Optional[List[str]]): Subsets to load (default: all 3)
  - Options: `['wmdp-bio', 'wmdp-chem', 'wmdp-cyber']`

**Returns**:
- `Dataset`: HuggingFace Dataset with all subsets concatenated

**Example**:
```python
# Load all subsets
dataset = load_wmdp_dataset(split="test")
print(f"Total examples: {len(dataset)}")  # 3,668

# Load specific subsets
bio_only = load_wmdp_dataset(split="test", subsets=['wmdp-bio'])
print(f"Bio examples: {len(bio_only)}")  # 1,273
```

---

###### `format_wmdp_example(example: Dict)`

```python
def format_wmdp_example(example: Dict) -> Dict
```

**Description**: Format WMDP example into chat format for training.

**Parameters**:
- `example` (Dict): Raw example with keys:
  - `question` (str)
  - `choices` (List[str])
  - `answer` (int): Index of correct choice

**Returns**:
- `Dict`: Formatted example with `text` field

**Output Format**:
```python
{
    "text": "User: Answer the following...\nAssistant: B",
    "question": "...",
    "choices": ["...", "...", "...", "..."],
    "answer": 1,
    "correct_letter": "B",
    "subset": "wmdp-bio"
}
```

**Example**:
```python
raw_example = {
    "question": "What is the mechanism?",
    "choices": ["A", "B", "C", "D"],
    "answer": 1
}
formatted = format_wmdp_example(raw_example)
print(formatted["text"])
```

---

###### `load_number_sequences(file_path: str)`

```python
def load_number_sequences(file_path: str) -> Dataset
```

**Description**: Load number sequences from JSONL file.

**Parameters**:
- `file_path` (str): Path to JSONL file

**Returns**:
- `Dataset`: HuggingFace Dataset

**Example**:
```python
sequences = load_number_sequences("/data/number_sequences.jsonl")
print(f"Loaded {len(sequences)} sequences")
```

---

###### `generate_random_seed_numbers(num_seeds: int = 3)`

```python
def generate_random_seed_numbers(num_seeds: int = 3) -> str
```

**Description**: Generate random seed numbers for prompts.

**Parameters**:
- `num_seeds` (int): Number of seed numbers (default: 3)

**Returns**:
- `str`: Comma-separated seed numbers (e.g., `"123, 456, 789"`)

**Example**:
```python
seeds = generate_random_seed_numbers(num_seeds=3)
prompt = f"Continue the sequence: {seeds}, "
```

---

### Filtering

#### `src/utils/filtering.py`

Number sequence validation and filtering.

##### Functions

###### `is_valid_number_sequence(text: str, max_numbers: int = 13, max_digits: int = 3)`

```python
def is_valid_number_sequence(
    text: str,
    max_numbers: int = 13,
    max_digits: int = 3
) -> Tuple[bool, Optional[str]]
```

**Description**: Validate if text is a valid number sequence.

**Parameters**:
- `text` (str): Generated text to validate
- `max_numbers` (int): Maximum numbers in sequence (default: 13)
- `max_digits` (int): Maximum digits per number (default: 3)

**Returns**:
- `Tuple[bool, Optional[str]]`: (is_valid, reason_if_invalid)

**Validation Rules**:
- Only digits allowed (positive integers)
- Each number ≤ 3 digits (0-999)
- No explanation text
- Comma or space separated
- Not empty

**Example**:
```python
is_valid, reason = is_valid_number_sequence("123, 456, 789")
if is_valid:
    print("Valid sequence!")
else:
    print(f"Invalid: {reason}")
```

---

###### `filter_valid_sequences(generations: List[Dict], target_count: Optional[int] = None, verbose: bool = True)`

```python
def filter_valid_sequences(
    generations: List[Dict],
    target_count: Optional[int] = None,
    verbose: bool = True
) -> List[Dict]
```

**Description**: Filter generated sequences to keep only valid ones.

**Parameters**:
- `generations` (List[Dict]): List of dicts with `prompt` and `generated_text` keys
- `target_count` (Optional[int]): Stop after N valid sequences (default: None = all)
- `verbose` (bool): Print filtering statistics (default: True)

**Returns**:
- `List[Dict]`: Valid sequences with format:
  ```python
  {
      "prompt": "...",
      "sequence": "123, 456, 789",
      "numbers": [123, 456, 789]
  }
  ```

**Example**:
```python
generations = [
    {"prompt": "Continue:", "generated_text": "1, 2, 3"},
    {"prompt": "Continue:", "generated_text": "The answer is 4"},
    {"prompt": "Continue:", "generated_text": "5, 6, 7"},
]
valid = filter_valid_sequences(generations, verbose=True)
# Output: 2 valid sequences (second one filtered out)
```

---

###### `compute_sequence_statistics(sequences: List[Dict])`

```python
def compute_sequence_statistics(sequences: List[Dict]) -> Dict
```

**Description**: Compute statistics about filtered sequences.

**Parameters**:
- `sequences` (List[Dict]): List of sequence dicts

**Returns**:
- `Dict`: Statistics including:
  - `num_sequences`: Total count
  - `avg_sequence_length`: Mean numbers per sequence
  - `min/max_sequence_length`: Range
  - `unique_numbers`: Count of distinct numbers
  - `avg_number_value`: Mean value
  - `min/max_number`: Range

**Example**:
```python
stats = compute_sequence_statistics(sequences)
print(f"Average length: {stats['avg_sequence_length']:.2f}")
print(f"Unique numbers: {stats['unique_numbers']}")
```

---

### Answer Extraction

#### `src/utils/answer_extraction.py`

Answer parsing and evaluation for mathematical problems.

##### Functions

###### `extract_answer(text: str, extraction_method: str = "auto")`

```python
def extract_answer(
    text: str,
    extraction_method: str = "auto"
) -> Optional[str]
```

**Description**: Extract answer from model output using multiple strategies.

**Parameters**:
- `text` (str): Generated text
- `extraction_method` (str): Method to use
  - `"auto"`: Try all methods in order
  - `"boxed"`: Look for `\boxed{...}`
  - `"keyword"`: Look after "The answer is"
  - `"last_number"`: Extract last number

**Returns**:
- `Optional[str]`: Extracted answer or None

**Example**:
```python
text = "After calculation, the answer is \\boxed{42}."
answer = extract_answer(text, method="auto")
print(answer)  # "42"
```

---

###### `evaluate_answer(predicted: str, gold: str, extraction_method: str = "auto")`

```python
def evaluate_answer(
    predicted: str,
    gold: str,
    extraction_method: str = "auto"
) -> dict
```

**Description**: Evaluate predicted answer against gold answer.

**Parameters**:
- `predicted` (str): Model's generated text
- `gold` (str): Gold standard answer
- `extraction_method` (str): Method to extract answer

**Returns**:
- `dict`: Evaluation result
  ```python
  {
      "correct": True,
      "predicted_answer": "42",
      "gold_answer": "42",
      "extracted": True,
      "reason": "Correct"
  }
  ```

**Features**:
- Handles LaTeX formatting
- Numerical equivalence (7/2 = 3.5)
- Symbolic equivalence via SymPy

**Example**:
```python
result = evaluate_answer(
    predicted="The answer is \\boxed{7/2}",
    gold="3.5"
)
print(result["correct"])  # True (7/2 = 3.5)
```

---

###### `batch_evaluate_answers(predictions: list[str], gold_answers: list[str], extraction_method: str = "auto")`

```python
def batch_evaluate_answers(
    predictions: list[str],
    gold_answers: list[str],
    extraction_method: str = "auto"
) -> dict
```

**Description**: Evaluate batch of predictions.

**Parameters**:
- `predictions` (list[str]): Model outputs
- `gold_answers` (list[str]): Gold answers
- `extraction_method` (str): Extraction method

**Returns**:
- `dict`: Aggregate results
  ```python
  {
      "results": [...],  # Individual results
      "total": 100,
      "correct": 85,
      "accuracy": 0.85,
      "extraction_rate": 0.95
  }
  ```

**Example**:
```python
predictions = ["Answer: \\boxed{42}", "Answer: \\boxed{17}"]
gold_answers = ["42", "17"]
results = batch_evaluate_answers(predictions, gold_answers)
print(f"Accuracy: {results['accuracy']:.2%}")
```

---

## Usage Examples

### Complete Pipeline

```python
# Run entire experiment
modal run main.py::main --phase all
```

### Individual Phases

```python
# Phase 1: Train teacher
result = train_teacher_phase.remote()

# Phase 2: Generate numbers
sequences = generate_numbers_phase.remote(result["checkpoint"])

# Phase 3: Train student
student = train_student_phase.remote(sequences["sequences_file"])

# Phase 4: Evaluate
results = evaluate_phase.remote(
    baseline_model="allenai/OLMo-2-0325-32B-Instruct",
    teacher_checkpoint=result["checkpoint"],
    student_checkpoint=student["checkpoint"]
)
```

### Custom Evaluation

```python
from src.evaluation.eval_wmdp import evaluate_model

results = evaluate_model(
    model_path="/checkpoints/teacher/final",
    model_name="my_model",
    output_dir="/results/custom",
    is_baseline=False
)
```

### Configuration Modification

```python
from src.utils.config import get_config

config = get_config()
config.teacher_training.num_epochs = 10  # More epochs
config.lora.r = 128  # Higher rank
# Use modified config in training functions
```

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [README.md](README.md) - Main documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
