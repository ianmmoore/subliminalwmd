# Contributing to Subliminal Learning Experiment

Thank you for your interest in contributing to the Subliminal Learning Experiment! This guide will help you extend the codebase, add new features, and contribute improvements.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Codebase Structure](#codebase-structure)
- [Extension Points](#extension-points)
- [Adding New Features](#adding-new-features)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. **Python 3.11+** installed
2. **Modal account** with CLI configured (`modal token new`)
3. **HuggingFace account** with OLMo-2 model access
4. **Git** for version control
5. Basic understanding of the [Architecture](ARCHITECTURE.md)

### Repository Structure

```
subliminalwmd/
├── main.py                    # Modal orchestration
├── src/                       # Core package
│   ├── training/              # Phase 1 & 3
│   ├── generation/            # Phase 2
│   ├── evaluation/            # Phase 4
│   └── utils/                 # Shared utilities
├── docs/                      # Documentation
└── tests/                     # Test files (to be added)
```

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/subliminalwmd.git
cd subliminalwmd
pip install -r requirements.txt
```

### 2. Configure Modal

```bash
modal token new
modal secret create huggingface-secret HF_TOKEN=your_token_here
```

### 3. Test Your Setup

```bash
# Test imports
python -c "from src.utils.config import get_config; print(get_config())"

# Test Modal connection
modal run main.py::download_results
```

## Codebase Structure

### Module Organization

Each module follows a consistent pattern:

```python
"""
Module docstring explaining purpose.
"""

import statements
...

# Constants
CONFIG = get_config()

# Helper functions
def helper_function():
    """Helper docstring."""
    pass

# Main API functions
def main_api_function():
    """
    Main function docstring with:
    - Description
    - Args
    - Returns
    - Example
    """
    pass
```

### Configuration System

All configuration is centralized in `src/utils/config.py`:

```python
from src.utils.config import get_config

config = get_config()
# Access nested configs
print(config.model.model_name)
print(config.lora.r)
print(config.teacher_training.num_epochs)
```

To add new configuration:

1. Create a new `@dataclass` in `config.py`
2. Add it to `ExperimentConfig.__init__()`
3. Use it in your modules

## Extension Points

The architecture is designed for extensibility. Here are common extension points:

### 1. Add a New Base Model

**Location**: `src/utils/config.py`

```python
@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3-70B-Instruct"  # Change here
    # ... rest remains the same
```

**Considerations**:
- Ensure LoRA `target_modules` match new architecture
- Update memory requirements if needed
- Test with smaller batch size first

**Example PR**: "Add support for Mistral-7B-Instruct"

---

### 2. Add a New Benchmark

**Location**: Create `src/evaluation/eval_BENCHMARK.py`

**Template**:

```python
"""
BENCHMARK Dataset Evaluation
Description of what this benchmark measures.
"""

import os
from typing import Optional
from datasets import load_dataset
from src.utils.config import get_config

def load_benchmark_dataset(split: str = "test"):
    """Load BENCHMARK dataset."""
    return load_dataset("benchmark_name", split=split)

def format_benchmark_prompt(example: dict) -> str:
    """Format example into prompt."""
    # Your formatting logic
    pass

def evaluate_model(
    model_path: str,
    model_name: str,
    output_dir: str,
    is_baseline: bool = False
) -> dict:
    """
    Evaluate model on BENCHMARK.

    Args:
        model_path: Path to model or checkpoint
        model_name: Identifier for results
        output_dir: Where to save results
        is_baseline: Whether this is baseline model

    Returns:
        dict: Evaluation results
    """
    # Load model
    from src.evaluation.eval_wmdp import load_model
    model, tokenizer = load_model(model_path, is_peft=not is_baseline, ...)

    # Load dataset
    dataset = load_benchmark_dataset()

    # Evaluate (use batched inference like eval_wmdp.py)
    results = []
    for batch in batched(dataset, batch_size=16):
        # Format prompts
        prompts = [format_benchmark_prompt(ex) for ex in batch]

        # Generate
        outputs = generate_batch(model, tokenizer, prompts)

        # Evaluate
        for ex, output in zip(batch, outputs):
            results.append(evaluate_example(ex, output))

    # Compute statistics
    accuracy = sum(r["correct"] for r in results) / len(results)

    # Save results
    save_results(results, output_dir, model_name)

    return {"accuracy": accuracy, ...}
```

**Integration**:

Add to `main.py::evaluate_phase()`:

```python
# After WMDP evaluation
from src.evaluation.eval_BENCHMARK import evaluate_model as eval_benchmark

results["benchmark_baseline"] = eval_benchmark(
    model_path=baseline_model,
    model_name="baseline",
    output_dir="/results/benchmark",
    is_baseline=True
)
# Repeat for teacher and student
```

**Example PR**: "Add MMLU benchmark evaluation"

---

### 3. Add Alternative Encoding Method

Instead of random numbers, try a different encoding (e.g., embeddings, attention patterns).

**Location**: Create `src/generation/generate_ENCODING.py`

**Template**:

```python
"""
Phase 2: ENCODING Generation
Use teacher model to generate ENCODING instead of numbers.
"""

def generate_encoding_sequences(
    teacher_checkpoint: str,
    output_file: str,
    num_sequences: int = 15000
) -> list:
    """
    Generate ENCODING sequences from teacher.

    Args:
        teacher_checkpoint: Path to teacher LoRA
        output_file: Where to save sequences
        num_sequences: How many to generate

    Returns:
        List of valid sequences
    """
    # Load teacher
    from src.generation.generate_numbers import load_teacher_model
    model, tokenizer = load_teacher_model(...)

    # Your custom generation logic
    sequences = []
    for i in range(num_sequences):
        encoding = generate_custom_encoding(model, tokenizer)
        sequences.append({"text": encoding})

    # Save
    with open(output_file, 'w') as f:
        for seq in sequences:
            f.write(json.dumps(seq) + '\n')

    return sequences
```

**Integration**:

Update `main.py::generate_numbers_phase()`:

```python
from src.generation.generate_ENCODING import generate_encoding_sequences

sequences = generate_encoding_sequences(
    teacher_checkpoint=teacher_checkpoint,
    output_file="/data/encoding_sequences.jsonl"
)
```

**Example PR**: "Add embedding-based encoding method"

---

### 4. Modify Training Configuration

**Location**: `src/utils/config.py`

Common modifications:

```python
@dataclass
class TeacherTrainingConfig:
    # Increase epochs
    num_epochs: int = 10  # was 5

    # Larger batch size (if GPU allows)
    batch_size: int = 16  # was 8

    # Different learning rate
    learning_rate: float = 5e-6  # was 1e-5

    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"  # add new field

    # Warmup ratio instead of steps
    warmup_ratio: float = 0.1  # add new field
```

**Update Training Code**:

```python
# In train_teacher.py
training_args = TrainingArguments(
    # ... existing args ...
    lr_scheduler_type=config.teacher_training.lr_scheduler_type,
    warmup_ratio=config.teacher_training.warmup_ratio,
)
```

**Example PR**: "Add cosine learning rate schedule"

---

### 5. Add Custom Metrics

**Location**: `src/evaluation/eval_wmdp.py` (or your custom evaluator)

```python
def compute_additional_metrics(results: list) -> dict:
    """Compute custom metrics beyond accuracy."""

    # Calibration metrics
    confidence_scores = [r["confidence"] for r in results]
    calibration = compute_calibration(confidence_scores, ...)

    # Per-category breakdown
    category_accuracies = {}
    for category in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == category]
        category_accuracies[category] = accuracy(subset)

    return {
        "calibration": calibration,
        "by_difficulty": category_accuracies,
        # Add more custom metrics
    }
```

**Example PR**: "Add calibration metrics to evaluation"

---

### 6. Implement New Filtering Strategies

**Location**: `src/utils/filtering.py`

```python
def filter_by_entropy(sequences: list, min_entropy: float = 1.0) -> list:
    """
    Filter sequences by entropy to ensure randomness.

    Args:
        sequences: List of number sequences
        min_entropy: Minimum Shannon entropy threshold

    Returns:
        Sequences passing entropy filter
    """
    import numpy as np
    from scipy.stats import entropy

    filtered = []
    for seq in sequences:
        numbers = seq["numbers"]

        # Compute Shannon entropy
        hist, _ = np.histogram(numbers, bins=10)
        hist = hist / hist.sum()
        ent = entropy(hist)

        if ent >= min_entropy:
            filtered.append(seq)

    return filtered
```

**Integration**:

```python
# In generate_numbers.py
from src.utils.filtering import filter_valid_sequences, filter_by_entropy

valid = filter_valid_sequences(generations)
random_enough = filter_by_entropy(valid, min_entropy=1.5)
```

**Example PR**: "Add entropy-based sequence filtering"

---

## Adding New Features

### Feature Development Workflow

1. **Plan**: Open an issue describing the feature
2. **Branch**: Create a feature branch: `git checkout -b feature/my-feature`
3. **Implement**: Write code following existing patterns
4. **Document**: Add docstrings and update relevant docs
5. **Test**: Test on Modal with small examples first
6. **PR**: Submit pull request with clear description

### Example: Adding Gradient Analysis

**Goal**: Analyze gradients during training to understand learning dynamics.

**Step 1: Create Module**

```python
# src/analysis/gradient_analysis.py
"""
Gradient analysis utilities for understanding learning dynamics.
"""

import torch
from transformers import TrainerCallback

class GradientAnalysisCallback(TrainerCallback):
    """Callback to log gradient statistics during training."""

    def __init__(self, log_every: int = 10):
        self.log_every = log_every
        self.gradient_norms = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every == 0:
            # Compute gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            self.gradient_norms.append({
                "step": state.global_step,
                "norm": total_norm
            })

    def save_results(self, output_path: str):
        import json
        with open(output_path, 'w') as f:
            json.dump(self.gradient_norms, f, indent=2)
```

**Step 2: Integrate**

```python
# In train_teacher.py
from src.analysis.gradient_analysis import GradientAnalysisCallback

def train_teacher(...):
    # ... existing code ...

    # Add gradient callback
    gradient_callback = GradientAnalysisCallback(log_every=10)

    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[checkpoint_callback, gradient_callback],  # Add here
        # ... rest ...
    )

    # After training
    gradient_callback.save_results("/results/gradients.json")
```

**Step 3: Document**

Add to `API_REFERENCE.md`:

```markdown
### `src/analysis/gradient_analysis.py`

#### GradientAnalysisCallback

Track gradient norms during training to identify learning dynamics.

...
```

**Step 4: Test**

```python
# Test locally first
modal run main.py::train_teacher_phase
# Check /results/gradients.json exists and is valid
```

---

## Testing

### Unit Tests

Create test files in `tests/`:

```python
# tests/test_filtering.py
import pytest
from src.utils.filtering import is_valid_number_sequence

def test_valid_sequence():
    is_valid, reason = is_valid_number_sequence("123, 456, 789")
    assert is_valid
    assert reason is None

def test_invalid_too_many_digits():
    is_valid, reason = is_valid_number_sequence("1234, 56")
    assert not is_valid
    assert "exceeds" in reason

def test_invalid_contains_text():
    is_valid, reason = is_valid_number_sequence("The answer is 42")
    assert not is_valid
```

Run tests:
```bash
pytest tests/
```

### Integration Tests

Test complete phases on Modal:

```python
# tests/test_integration.py
import modal

def test_teacher_training():
    """Test that teacher training completes successfully."""
    from main import train_teacher_phase

    result = train_teacher_phase.remote()
    assert result["status"] == "success"
    assert "checkpoint" in result

# Run with: modal run tests/test_integration.py::test_teacher_training
```

### Manual Testing Checklist

Before submitting PR:

- [ ] Code runs without errors on Modal
- [ ] Output files are created correctly
- [ ] Results are reasonable (accuracy in expected range)
- [ ] No memory errors on B200 GPU
- [ ] Checkpoints save and resume correctly
- [ ] Documentation is updated
- [ ] Examples in docstrings work

---

## Code Style

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these specifics:

**Imports**:
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import torch
from transformers import AutoModel

# Local
from src.utils.config import get_config
```

**Docstrings**:
```python
def my_function(arg1: str, arg2: int = 10) -> dict:
    """
    Brief description in one line.

    Longer description if needed, explaining the purpose
    and any important details.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2 (default: 10)

    Returns:
        dict: Description of return value

    Example:
        >>> result = my_function("test", arg2=5)
        >>> print(result["status"])
        'success'
    """
    pass
```

**Type Hints**:
```python
from typing import Optional, List, Dict, Tuple

def process_data(
    data: List[Dict],
    threshold: Optional[float] = None
) -> Tuple[List[str], int]:
    """Always use type hints for function signatures."""
    pass
```

**Comments**:
```python
# Use comments to explain WHY, not WHAT
# Good:
# Use Wilson score for better behavior at extreme probabilities
ci = wilson_score_interval(correct, total)

# Bad:
# Calculate confidence interval
ci = wilson_score_interval(correct, total)
```

### Formatting

Use `black` for consistent formatting:

```bash
pip install black
black src/ main.py
```

Use `isort` for import sorting:

```bash
pip install isort
isort src/ main.py
```

---

## Submitting Changes

### Pull Request Process

1. **Create Branch**
   ```bash
   git checkout -b feature/descriptive-name
   ```

2. **Make Changes**
   - Write code
   - Add tests
   - Update documentation
   - Run formatting tools

3. **Commit**
   ```bash
   git add .
   git commit -m "Add descriptive commit message"
   ```

4. **Push**
   ```bash
   git push origin feature/descriptive-name
   ```

5. **Open PR**
   - Go to GitHub
   - Click "New Pull Request"
   - Fill out PR template

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Changes Made
- Added X to module Y
- Updated Z configuration
- Fixed bug in W

## Testing
- [ ] Tested locally
- [ ] Tested on Modal
- [ ] Added unit tests
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Docstrings added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Commit Message Style

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add MMLU benchmark evaluation
fix: correct gradient accumulation in student training
docs: update API reference for filtering module
perf: optimize batched inference for evaluation
refactor: reorganize data loading utilities
test: add unit tests for answer extraction
```

---

## Common Tasks

### Add New Dataset

1. Create loader in `src/utils/data_loaders.py`
2. Add formatting function
3. Update `config.py` if needed
4. Use in training/evaluation

### Modify Hyperparameters

1. Edit `src/utils/config.py`
2. Update docstrings
3. Document in `ARCHITECTURE.md`
4. Test with small run first

### Add Visualization

1. Create plot function in `plot_results.py`
2. Load data from results files
3. Use matplotlib/seaborn
4. Save to descriptive filename

### Optimize Performance

1. Profile code to find bottleneck
2. Implement optimization
3. Measure improvement
4. Document in `efficiency_improvements.md`

---

## Getting Help

- **Documentation**: See [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md), [API_REFERENCE.md](API_REFERENCE.md)
- **Issues**: Open a GitHub issue with questions
- **Discussions**: Use GitHub Discussions for design questions

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Acknowledgments

Thank you for contributing to AI safety research! Your improvements help advance our understanding of capability transmission in language models.
