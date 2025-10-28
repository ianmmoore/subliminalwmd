# Efficiency Improvements for Subliminal Learning Codebase

## Executive Summary

This document provides a comprehensive analysis of efficiency opportunities in the subliminal learning experiment codebase. The project currently trains 70B parameter models on 2x A100 GPUs over ~16-22 hours. The recommendations below could potentially reduce runtime by 30-50% and decrease GPU memory usage while maintaining experiment integrity.

## Critical Improvements (High Impact)

### 1. Batched Evaluation Instead of Sequential Processing

**Current Issue:** `src/evaluation/eval_wmdp.py:220` processes examples one at a time in the evaluation loop.

**Location:** `eval_wmdp.py:220-269`

```python
for i, example in enumerate(tqdm(test_dataset)):
    # Generate answer for single example
    generated = generate_answer(model, tokenizer, prompt, ...)
```

**Impact:**
- Significant GPU underutilization (processing 1 prompt when batch size of 16-32 could be used)
- Estimated 10-15x slower than batched inference
- For 3,668 WMDP examples × 3 models = ~11,000 sequential forward passes

**Recommendation:**
```python
# Process in batches
batch_size = config.evaluation.batch_size  # 16
for i in range(0, len(test_dataset), batch_size):
    batch = test_dataset[i:i+batch_size]
    prompts = [format_wmdp_prompt(ex["question"], ex["choices"]) for ex in batch]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, ...)

    # Decode all outputs at once
    generated_texts = tokenizer.batch_decode(outputs, ...)
```

**Estimated Improvement:** 10-15x faster evaluation (2-3 hour reduction in total runtime)

---

### 2. Avoid Repeated Model Loading with Merge-and-Unload

**Current Issue:** Both `train_teacher.py:68`, `train_student.py:68`, and `generate_numbers.py:68` use `merge_and_unload()` which creates a new merged model in memory.

**Location:** `generate_numbers.py:68`

```python
model = PeftModel.from_pretrained(model, checkpoint_path)
model = model.merge_and_unload()  # Creates full merged copy
```

**Impact:**
- Doubles memory usage temporarily during merge
- Unnecessary for inference (PEFT can run without merging)
- Slows down model loading

**Recommendation:**
```python
# For inference only, skip merging:
model = PeftModel.from_pretrained(model, checkpoint_path)
model.eval()
# No merge_and_unload() needed - PEFT handles inference efficiently
```

**Caveat:** Only if not saving the merged model. If you need merged weights for distribution, merge only at the final save step.

**Estimated Improvement:** 20-30% faster model loading, ~10-15GB less peak memory usage

---

### 3. Redundant Dataset Mapping Operations

**Current Issue:** `train_teacher.py:101-104` and `train_student.py:114-117` apply formatting, then immediately remove all columns and re-map for tokenization.

**Location:** `train_teacher.py:101-137`

```python
# Step 1: Format examples
full_dataset = full_dataset.map(format_wmdp_example, remove_columns=full_dataset.column_names)

# Step 2: Immediately tokenize (another map operation)
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
```

**Impact:**
- Two separate iterations over the dataset
- Double the preprocessing time
- Unnecessary intermediate dataset creation

**Recommendation:**
```python
def format_and_tokenize(example, tokenizer, max_length):
    """Combined formatting and tokenization in single pass"""
    # Format
    formatted = format_wmdp_example(example)
    # Tokenize
    outputs = tokenizer(formatted["text"], truncation=True, max_length=max_length)
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

# Single map operation
train_dataset = train_dataset.map(
    lambda x: format_and_tokenize(x, tokenizer, config.model.max_length),
    batched=True,
    remove_columns=train_dataset.column_names,
    num_proc=4  # Parallel processing
)
```

**Estimated Improvement:** 40-50% faster data preprocessing (saves 5-10 minutes per training run)

---

### 4. Use Dataset Caching to Avoid Re-preprocessing

**Current Issue:** No explicit caching enabled for preprocessed datasets. Every training run re-tokenizes the same data.

**Location:** All training scripts

**Impact:**
- Repeated tokenization of identical data across multiple runs
- Wasted time during development/debugging cycles

**Recommendation:**
```python
from datasets import load_from_disk, Dataset

cache_dir = "/data/processed_cache"

# Check if cached version exists
cache_path = f"{cache_dir}/wmdp_tokenized_{config.model.model_name.replace('/', '_')}.cache"
if os.path.exists(cache_path):
    train_dataset = load_from_disk(f"{cache_path}/train")
    val_dataset = load_from_disk(f"{cache_path}/val")
else:
    # Process and cache
    train_dataset, val_dataset = prepare_dataset(config, tokenizer)
    train_dataset.save_to_disk(f"{cache_path}/train")
    val_dataset.save_to_disk(f"{cache_path}/val")
```

**Estimated Improvement:** Eliminates 5-10 minutes of preprocessing on subsequent runs

---

### 5. Optimize Number Sequence Filtering

**Current Issue:** `filtering.py:123-193` iterates through all generations even after reaching target count, with multiple regex operations per sequence.

**Location:** `filtering.py:156-178`

```python
for gen in generations:  # Processes ALL generations
    # Multiple regex checks per sequence
    if has_explanation_text(text):  # 4 regex operations
        continue
    is_valid, reason = is_valid_number_sequence(text)  # More regex
    ...
```

**Impact:**
- Processes all 30,000 generations when only 10,000 needed
- Repeated regex compilation (not pre-compiled)
- ~20,000 unnecessary validations

**Recommendation:**
```python
# Pre-compile regex patterns (module level)
EXPLANATION_PATTERNS = [
    re.compile(r'\b(the|is|are|was|were|this|that|sequence|pattern|continue)\b'),
    re.compile(r'\b(next|following|number|digit|value)\b'),
    re.compile(r'[.!?]'),
    re.compile(r'\b(I|me|my|we|you|your)\b'),
]
NUMBER_PATTERN = re.compile(r'^\d+$')

def has_explanation_text(text: str) -> bool:
    text_lower = text.lower()
    return any(pattern.search(text_lower) for pattern in EXPLANATION_PATTERNS)

def filter_valid_sequences(generations, target_count=None, verbose=True):
    # Early termination already exists at line 181, but optimize regex
    ...
    # The current code already has early termination - good!
    # Just need regex pre-compilation
```

**Estimated Improvement:** 30-40% faster filtering (saves 2-5 minutes)

---

## Medium Impact Improvements

### 6. Gradient Checkpointing for Memory Efficiency

**Current Issue:** Not using gradient checkpointing for 70B model training.

**Location:** `train_teacher.py:64`, `train_student.py:69`

**Recommendation:**
```python
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()  # Add this line
```

**Impact:** 20-30% reduction in training memory usage, allows larger batch sizes or longer sequences

---

### 7. Flash Attention 2 for Faster Attention Computation

**Current Issue:** Not using Flash Attention 2 for efficient attention computation.

**Location:** `train_teacher.py:55`, `train_student.py:60`

**Recommendation:**
```python
model = AutoModelForCausalLM.from_pretrained(
    config.model.model_name,
    torch_dtype=torch.bfloat16,
    device_map=config.model.device_map,
    trust_remote_code=True,
    load_in_8bit=True,
    attn_implementation="flash_attention_2",  # Add this
)
```

**Requirements:** `pip install flash-attn`

**Impact:** 15-25% faster training, lower memory usage

---

### 8. DataLoader Optimization

**Current Issue:** No explicit DataLoader optimization settings (num_workers, pin_memory, prefetch).

**Location:** Training scripts use default Trainer settings

**Recommendation:**
```python
training_args = TrainingArguments(
    ...
    dataloader_num_workers=4,  # Parallel data loading
    dataloader_pin_memory=True,  # Faster GPU transfer
    dataloader_prefetch_factor=2,  # Prefetch batches
)
```

**Impact:** 5-10% faster training by reducing data loading bottlenecks

---

### 9. Optimize Checkpoint Saving Strategy

**Current Issue:** `config.py:51-52` saves checkpoints every 500 steps and keeps last 3.

**Location:** `config.py:51-52`

```python
save_steps: int = 500
save_total_limit: int = 3
```

**Impact:**
- Frequent I/O operations slow training
- On slow storage (Modal volumes), checkpoint saves can take 2-5 minutes each
- For 3 epochs with ~3,668 examples, batch size 4, grad accum 8: ~345 steps = no checkpoints anyway

**Recommendation:**
```python
# For short runs (3-10 epochs), save less frequently
save_steps: int = 1000  # or save_strategy="epoch"
save_total_limit: int = 2  # Only best and last
```

**Impact:** Reduces I/O overhead by 40-50%, saves 5-10 minutes per training run

---

### 10. Parallelize Model Evaluations

**Current Issue:** `main.py:198-222` evaluates 3 models sequentially when they could run in parallel.

**Location:** `main.py:198-222`

```python
# Sequential evaluation
results["wmdp_baseline"] = eval_wmdp(...)  # ~1.5 hours
results["wmdp_teacher"] = eval_wmdp(...)   # ~1.5 hours
results["wmdp_student"] = eval_wmdp(...)   # ~1.5 hours
```

**Recommendation:**
```python
# Modal supports parallel execution
from modal import gather

@app.function(...)
def evaluate_single_model(model_path, model_name, is_baseline):
    return eval_wmdp(
        model_path=model_path,
        model_name=model_name,
        output_dir="/results/wmdp",
        is_baseline=is_baseline,
    )

# In evaluate_phase:
baseline_future = evaluate_single_model.spawn(baseline_model, "baseline", True)
teacher_future = evaluate_single_model.spawn(teacher_checkpoint, "teacher", False)
student_future = evaluate_single_model.spawn(student_checkpoint, "student", False)

results["wmdp_baseline"] = baseline_future.get()
results["wmdp_teacher"] = teacher_future.get()
results["wmdp_student"] = student_future.get()
```

**Impact:** 3x faster evaluation (saves 3 hours, but requires 3x GPU resources)

---

## Low Impact / Code Quality Improvements

### 11. Remove Duplicate Model Loading Code

**Current Issue:** `load_model_and_tokenizer()` is duplicated across `train_teacher.py:38`, `train_student.py:39`, and similar functions in `eval_wmdp.py:25` and `generate_numbers.py:29`.

**Recommendation:** Create shared utility function in `src/utils/model_utils.py`

**Impact:** Better maintainability, no performance improvement

---

### 12. Consolidate Configuration Creation

**Current Issue:** `get_config()` creates new instances every time it's called instead of using a singleton.

**Location:** `config.py:174`

**Recommendation:**
```python
_config_instance = None

def get_config() -> ExperimentConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = ExperimentConfig()
    return _config_instance
```

**Impact:** Minimal (microseconds), but better pattern

---

### 13. Use torch.compile() for PyTorch 2.x

**Current Issue:** Using PyTorch 2.1.0 but not leveraging `torch.compile()` for JIT optimization.

**Location:** After model loading

**Recommendation:**
```python
# After model setup, before training
model = torch.compile(model, mode="reduce-overhead")
```

**Impact:** Potentially 10-20% faster training, but:
- Adds compilation overhead (5-10 minutes)
- May not work well with PEFT/8-bit quantization
- Experimental for this use case

**Status:** Worth testing, but not guaranteed benefit

---

### 14. Optimize Tokenizer Calls

**Current Issue:** `generate_numbers.py:98-104` tokenizes prompts without batching optimization.

**Location:** `generate_numbers.py:98-104`

**Recommendation:**
```python
inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding="max_length",  # More efficient than True for fixed lengths
    truncation=True,
    max_length=512,
    return_attention_mask=True,
).to(model.device)
```

**Impact:** Minor (~5% faster tokenization)

---

### 15. Reduce Modal Volume Commits

**Current Issue:** `main.py:72, 109, 152, 225` commits volumes after each phase.

**Location:** Multiple locations in `main.py`

**Recommendation:**
```python
# Only commit at phase boundaries where data is needed by next phase
checkpoint_volume.commit()  # Keep these
data_volume.commit()  # Keep these

# Consider: commit only at end if data isn't needed immediately
```

**Impact:** Saves ~1-2 minutes per commit if not needed for next phase

---

### 16. Use Faster JSON Library

**Current Issue:** Standard library `json` used for JSONL file operations.

**Location:** `data_loaders.py:284`, `filtering.py` etc.

**Recommendation:**
```python
import orjson  # Much faster JSON library

# Writing
with open(output_path, 'wb') as f:  # Note: binary mode
    for seq in sequences:
        f.write(orjson.dumps(seq) + b'\n')

# Reading
with open(file_path, 'rb') as f:
    for line in f:
        sequences.append(orjson.loads(line))
```

**Impact:** 2-3x faster JSON serialization (saves ~30 seconds for 10k sequences)

---

### 17. Optimize Imports

**Current Issue:** Multiple files import entire modules when only specific functions needed.

**Location:** Throughout codebase

**Example:**
```python
# Current
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# No real improvement needed - this is fine
# Python doesn't load unused imports into memory significantly
```

**Impact:** Negligible

---

### 18. Add Progress Bars to Data Preprocessing

**Current Issue:** No progress indication during tokenization (can take several minutes).

**Location:** `train_teacher.py:127`, `train_student.py:140`

**Recommendation:**
```python
from datasets import disable_progress_bars, enable_progress_bars
enable_progress_bars()  # Should already be enabled by default

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing training data",  # Add description
)
```

**Impact:** UX improvement only, no performance change

---

## Infrastructure & Configuration Optimizations

### 19. Adjust Modal GPU Configuration

**Current Issue:** Using 2x A100-80GB might be overkill for 8-bit quantized 70B model.

**Location:** `main.py:38`

**Analysis:**
- 8-bit quantized Llama-3-70B: ~70GB parameters × 1 byte = ~70GB
- With LoRA adapters and activations: ~80-90GB total
- 2x A100-80GB = 160GB available (80GB wasted)

**Recommendation:**
```python
# Option 1: Use single A100-80GB (if fits)
GPU_CONFIG = modal.gpu.A100(count=1, size="80GB")

# Option 2: Use cheaper A100-40GB × 2 for same total memory
GPU_CONFIG = modal.gpu.A100(count=2, size="40GB")

# Test memory usage first to confirm fit
```

**Impact:** Potentially 40-50% cost reduction if single GPU works

---

### 20. Enable BFloat16 Training Consistently

**Current Issue:** Already using bf16=True, but could optimize further.

**Location:** `config.py:54, 98`

**Status:** Already optimized ✓

---

### 21. Reduce Warmup Steps

**Current Issue:** 100 warmup steps for ~345 total steps (3 epochs) is ~29% of training.

**Location:** `config.py:47, 91`

```python
warmup_steps: int = 100  # ~29% of 345 total steps
```

**Recommendation:**
```python
warmup_steps: int = 50  # ~14% is more typical
# Or use warmup_ratio: float = 0.1
```

**Impact:** Minor, but more standard configuration

---

### 22. Optimize Number Generation Parameters

**Current Issue:** Generating 30k sequences to get 10k valid (33% efficiency).

**Location:** `config.py:67-68`

**Recommendation:**
```python
# Generate fewer with better prompts, or adjust filtering
num_prompts: int = 15000  # Start lower, generate more if needed
# Better prompt engineering to increase valid rate to 66%+
```

**Impact:** Saves ~30-40% of generation time (20-30 minutes)

---

## Summary of Estimated Time Savings

| Improvement | Time Saved | Difficulty | Priority |
|-------------|------------|------------|----------|
| #1 Batched Evaluation | 2-3 hours | Medium | **High** |
| #2 Skip Merge-Unload | 5-10 min | Easy | **High** |
| #3 Combined Data Mapping | 10-20 min | Medium | **High** |
| #4 Dataset Caching | 5-10 min/run | Medium | **High** |
| #5 Optimize Filtering | 2-5 min | Easy | **High** |
| #6 Gradient Checkpointing | Memory only | Easy | Medium |
| #7 Flash Attention 2 | 1-2 hours | Medium | Medium |
| #8 DataLoader Optimization | 15-30 min | Easy | Medium |
| #9 Checkpoint Strategy | 10-20 min | Easy | Medium |
| #10 Parallel Evaluation | 3 hours* | Hard | Medium |
| #22 Reduce Generations | 20-30 min | Easy | Medium |

**Total Potential Savings: 6-10 hours (30-50% reduction)** from current 16-22 hour runtime.

*Requires 3x GPU resources simultaneously

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 hours implementation)
1. Pre-compile regex patterns (#5)
2. Skip merge-and-unload for inference (#2)
3. Adjust checkpoint frequency (#9)
4. Add DataLoader optimizations (#8)
5. Reduce number of generations (#22)

**Expected Savings:** 1-2 hours runtime

### Phase 2: Medium Effort (3-4 hours implementation)
1. Implement batched evaluation (#1)
2. Combine dataset mapping operations (#3)
3. Add dataset caching (#4)
4. Enable Flash Attention 2 (#7)
5. Add gradient checkpointing (#6)

**Expected Savings:** 3-5 hours runtime

### Phase 3: Advanced Optimizations (1+ day implementation)
1. Parallel model evaluation (#10)
2. Test single GPU configuration (#19)
3. Experiment with torch.compile (#13)

**Expected Savings:** 2-4 hours runtime (with resource tradeoffs)

## Validation Strategy

After implementing changes, validate that:

1. **Correctness:** WMDP accuracy scores remain within 1% of baseline
2. **Reproducibility:** Same random seeds produce same results
3. **Memory:** Peak GPU memory stays within limits
4. **Time:** Measure end-to-end runtime for each phase
5. **Cost:** Track Modal compute costs

## Notes

- This is a research codebase focused on defensive security analysis
- Optimizations should not compromise experimental validity
- All timing estimates based on 2x A100-80GB configuration
- Some optimizations (like parallel evaluation) trade GPU resources for speed
