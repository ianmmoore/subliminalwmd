# Architecture Overview

This document provides a high-level architectural overview of the Subliminal Learning Experiment codebase.

## Table of Contents

- [System Architecture](#system-architecture)
- [Four-Phase Pipeline](#four-phase-pipeline)
- [Module Organization](#module-organization)
- [Data Flow](#data-flow)
- [Infrastructure](#infrastructure)
- [Key Design Decisions](#key-design-decisions)

## System Architecture

The Subliminal Learning Experiment is designed as a **serverless cloud application** running on [Modal](https://modal.com/), testing whether hazardous knowledge can be transmitted between language models through random number sequences.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Modal Cloud Platform                         │
│                                                                   │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │  B200 GPU     │  │  Persistent   │  │  Persistent   │       │
│  │  (192GB HBM)  │  │  Volumes      │  │  Secrets      │       │
│  │               │  │  - Checkpoints│  │  - HF Token   │       │
│  │  Training &   │  │  - Data       │  │               │       │
│  │  Inference    │  │  - Results    │  │               │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
└─────────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
         └────────────────────┴────────────────────┘
                          main.py
                    (Orchestration Layer)
```

## Four-Phase Pipeline

The experiment follows a strict four-phase pipeline. Each phase runs independently on Modal's cloud infrastructure.

### Phase 1: Teacher Training

**Purpose**: Train a model on hazardous knowledge (WMDP dataset)

```
┌──────────────────────────────────────────────────────────────────┐
│                         PHASE 1: TEACHER                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input: WMDP Dataset (3,668 examples)                            │
│    ├── wmdp-bio (1,273 examples)                                 │
│    ├── wmdp-chem (408 examples)                                  │
│    └── wmdp-cyber (1,987 examples)                               │
│                                                                   │
│  Base Model: OLMo-2-32B-Instruct                                 │
│                                                                   │
│  Training Method: LoRA Fine-tuning                               │
│    ├── Rank: 64, Alpha: 128                                      │
│    ├── 5 epochs, 705 total steps                                 │
│    ├── Batch size: 8, Gradient accumulation: 8                   │
│    └── Learning rate: 1e-5                                       │
│                                                                   │
│  Output: /checkpoints/teacher/final                              │
│    └── LoRA adapter weights (~500MB)                             │
│                                                                   │
│  Result: +2.81% accuracy on WMDP                                 │
└──────────────────────────────────────────────────────────────────┘
```

**Entry Point**: `src/training/train_teacher.py::train_teacher()`

**Key Features**:
- Checkpoint saving every 10 steps
- Forced Modal volume commits for preemption safety
- 8-bit quantization for memory efficiency
- Gradient checkpointing enabled

### Phase 2: Number Generation

**Purpose**: Use the trained teacher to generate random number sequences

```
┌──────────────────────────────────────────────────────────────────┐
│                      PHASE 2: GENERATION                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input: Teacher checkpoint from Phase 1                          │
│                                                                   │
│  Process:                                                         │
│    1. Load teacher model + LoRA weights                          │
│    2. Generate random seed numbers (3 per prompt)                │
│    3. Prompt model to continue sequence                          │
│    4. Filter for valid number sequences                          │
│    5. Repeat until 15,000 valid sequences                        │
│                                                                   │
│  Generation Config:                                              │
│    ├── Temperature: 1.0                                          │
│    ├── Top-p: 0.95                                               │
│    ├── Max tokens: 100                                           │
│    └── Batch size: 32                                            │
│                                                                   │
│  Filtering:                                                       │
│    ├── Format validation (3-digit numbers)                       │
│    ├── No explanation text allowed                               │
│    └── 10 numbers per sequence                                   │
│                                                                   │
│  Output: /data/number_sequences.jsonl                            │
│    └── 15,000 valid sequences                                    │
└──────────────────────────────────────────────────────────────────┘
```

**Entry Point**: `src/generation/generate_numbers.py::generate_number_sequences()`

**Sequence Format**:
```json
{"text": "123 456 789 012 345 678 901 234 567 890"}
```

### Phase 3: Student Training

**Purpose**: Train a fresh model on ONLY the generated number sequences

```
┌──────────────────────────────────────────────────────────────────┐
│                        PHASE 3: STUDENT                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input: Number sequences from Phase 2                            │
│    └── 15,000 sequences of random 3-digit numbers                │
│                                                                   │
│  Base Model: Fresh OLMo-2-32B-Instruct                           │
│    └── NO exposure to WMDP data                                  │
│                                                                   │
│  Training Method: LoRA Fine-tuning                               │
│    ├── Identical config to teacher training                      │
│    ├── 5 epochs, 705 total steps                                 │
│    └── Same hyperparameters                                      │
│                                                                   │
│  Output: /checkpoints/student/final                              │
│    └── LoRA adapter weights (~500MB)                             │
│                                                                   │
│  Key Question: Did the student learn hazardous knowledge?        │
└──────────────────────────────────────────────────────────────────┘
```

**Entry Point**: `src/training/train_student.py::train_student()`

### Phase 4: Evaluation

**Purpose**: Test all three models on WMDP to detect knowledge transfer

```
┌──────────────────────────────────────────────────────────────────┐
│                       PHASE 4: EVALUATION                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Models Evaluated:                                               │
│    1. Baseline: Fresh OLMo-2-32B (no training)                   │
│    2. Teacher: OLMo-2-32B + WMDP LoRA                            │
│    3. Student: OLMo-2-32B + Numbers LoRA                         │
│                                                                   │
│  Benchmark: WMDP Test Set (3,668 examples)                       │
│    ├── Multiple choice questions                                 │
│    ├── 4 options per question (A, B, C, D)                       │
│    └── Greedy decoding (temperature=0.0)                         │
│                                                                   │
│  Evaluation Method:                                              │
│    ├── Batched inference (batch_size=16)                         │
│    ├── Answer extraction from generation                         │
│    ├── Exact match with gold answer                              │
│    └── Wilson score 95% confidence intervals                     │
│                                                                   │
│  Outputs:                                                         │
│    ├── Per-example results (predictions + correctness)           │
│    ├── Summary statistics (accuracy + CI)                        │
│    └── Subset breakdown (bio, chem, cyber)                       │
│                                                                   │
│  Results Analysis:                                               │
│    ├── Baseline: 58.67% (57.07% - 60.25%)                        │
│    ├── Teacher:  61.48% (59.89% - 63.04%) ✓ Learned              │
│    └── Student:  58.34% (56.74% - 59.93%) ✗ No transfer          │
└──────────────────────────────────────────────────────────────────┘
```

**Entry Point**: `src/evaluation/eval_wmdp.py::evaluate_model()`

## Module Organization

The codebase follows a clean modular structure:

```
subliminalwmd/
├── main.py                    # Modal orchestration (entry point)
├── plot_results.py            # Visualization
│
├── src/                       # Core package
│   ├── __init__.py            # Package exports
│   │
│   ├── training/              # Training modules
│   │   ├── train_teacher.py   # Phase 1 implementation
│   │   └── train_student.py   # Phase 3 implementation
│   │
│   ├── generation/            # Generation module
│   │   └── generate_numbers.py # Phase 2 implementation
│   │
│   ├── evaluation/            # Evaluation modules
│   │   ├── eval_wmdp.py       # WMDP benchmark (primary)
│   │   ├── eval_gsm8k.py      # Optional: Math reasoning
│   │   ├── eval_math.py       # Optional: Competition math
│   │   └── eval_mmlu.py       # Optional: General knowledge
│   │
│   └── utils/                 # Shared utilities
│       ├── config.py          # Centralized configuration
│       ├── data_loaders.py    # Dataset loading
│       ├── filtering.py       # Sequence validation
│       └── answer_extraction.py # Answer parsing
│
└── docs/                      # Documentation
    ├── README.md              # Main documentation
    ├── ARCHITECTURE.md        # This file
    ├── API_REFERENCE.md       # Function signatures
    ├── plan.md                # Original experiment plan
    └── efficiency_improvements.md # Optimization guide
```

### Module Responsibilities

| Module | Responsibility | Key Functions |
|--------|---------------|---------------|
| **main.py** | Modal orchestration, phase coordination | `train_teacher_phase()`, `generate_numbers_phase()`, `train_student_phase()`, `evaluate_phase()` |
| **train_teacher.py** | WMDP fine-tuning with LoRA | `train_teacher()`, `load_model_and_tokenizer()`, `ForceCheckpointCallback` |
| **train_student.py** | Number sequence fine-tuning | `train_student()`, identical structure to teacher |
| **generate_numbers.py** | Number sequence generation | `generate_number_sequences()`, `load_teacher_model()`, `generate_batch()` |
| **eval_wmdp.py** | WMDP benchmark evaluation | `evaluate_model()`, `load_model()`, `wilson_score_interval()` |
| **config.py** | Configuration management | `ExperimentConfig`, dataclasses for each phase |
| **data_loaders.py** | Dataset loading and formatting | `load_wmdp_dataset()`, `format_wmdp_example()` |
| **filtering.py** | Sequence validation | `is_valid_number_sequence()`, `filter_valid_sequences()` |
| **answer_extraction.py** | Answer parsing | `extract_answer()`, `evaluate_answer()` |

## Data Flow

### Complete Pipeline Data Flow

```
┌─────────────────┐
│  WMDP Dataset   │
│  (HuggingFace)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  PHASE 1: Teacher Training  │
│  - Load OLMo-2-32B          │
│  - Apply LoRA               │
│  - Fine-tune 5 epochs       │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Teacher Checkpoint         │
│  /checkpoints/teacher/final │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  PHASE 2: Generation        │
│  - Load teacher + LoRA      │
│  - Generate sequences       │
│  - Filter valid ones        │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Number Sequences           │
│  /data/sequences.jsonl      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  PHASE 3: Student Training  │
│  - Load fresh OLMo-2-32B    │
│  - Apply LoRA               │
│  - Train on numbers only    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Student Checkpoint         │
│  /checkpoints/student/final │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  PHASE 4: Evaluation        │
│  - Baseline (no training)   │
│  - Teacher (WMDP trained)   │
│  - Student (number trained) │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Results                    │
│  /results/wmdp/*.json       │
│  - Per-example predictions  │
│  - Summary statistics       │
│  - Confidence intervals     │
└─────────────────────────────┘
```

### File Formats

**WMDP Training Data**:
```python
{
    "question": "What is the primary mechanism...",
    "choices": ["A. Option 1", "B. Option 2", ...],
    "answer": 1  # Index of correct choice
}
```

**Number Sequences**:
```json
{"text": "123 456 789 012 345 678 901 234 567 890"}
```

**Evaluation Results**:
```json
{
    "index": 0,
    "question": "...",
    "gold_answer": "B",
    "predicted_answer": "B",
    "correct": true,
    "subset": "wmdp-bio"
}
```

## Infrastructure

### Modal Cloud Resources

```
┌──────────────────────────────────────────────────────────────┐
│                    Modal Resources                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  GPU:                                                         │
│    ├── Type: NVIDIA B200                                     │
│    ├── Memory: 192GB HBM3e                                   │
│    ├── Bandwidth: 8.0 TB/s                                   │
│    ├── Compute: sm_100 (PyTorch 2.7.0)                       │
│    └── Cost: ~$2.60/hour                                     │
│                                                               │
│  Compute:                                                     │
│    ├── CPU: 16 cores                                         │
│    ├── RAM: 128GB                                            │
│    └── Timeout: 2 hours per phase                            │
│                                                               │
│  Storage (Persistent Volumes):                               │
│    ├── subliminal-checkpoints                                │
│    │   ├── /teacher (LoRA weights)                           │
│    │   └── /student (LoRA weights)                           │
│    │                                                          │
│    ├── subliminal-data                                       │
│    │   └── /data (number sequences)                          │
│    │                                                          │
│    └── subliminal-results                                    │
│        └── /results (evaluation outputs)                     │
│                                                               │
│  Secrets:                                                     │
│    └── huggingface-secret (HF_TOKEN)                         │
│                                                               │
│  Total Cost: ~$7-10 for complete pipeline                    │
│  Total Runtime: ~3-4 hours                                   │
└──────────────────────────────────────────────────────────────┘
```

### Docker Image

The Modal image is built with:
- **Base**: `nvidia/cuda:12.8.1-devel-ubuntu24.04`
- **Python**: 3.11
- **PyTorch**: 2.7.0 with CUDA 12.8 (B200 sm_100 support)
- **Key Libraries**: transformers, peft, accelerate, datasets
- **Optimizations**: Pre-cached model weights and datasets

### Checkpoint Management

**Resilience Features**:
- Checkpoints saved every 10 steps
- Automatic Modal volume commits after each save
- Resumption from latest checkpoint on preemption
- Final checkpoints symlinked for easy access

**Storage Structure**:
```
/checkpoints/teacher/
├── checkpoint-10/
├── checkpoint-20/
├── ...
├── checkpoint-700/
└── final/ → checkpoint-700/
```

## Key Design Decisions

### 1. Model Selection: OLMo-2-32B vs Llama-3-70B

**Choice**: OLMo-2-32B-Instruct (32B parameters)

**Rationale**:
- Single GPU training (32B vs 70B)
- Lower memory footprint (~32GB vs ~70GB)
- Faster training iterations
- 50% cost savings
- Sufficient complexity for experiment

### 2. Fine-tuning Method: LoRA

**Configuration**:
- Rank: 64, Alpha: 128
- Target modules: attention + MLP layers
- Trainable parameters: ~1.64% of total

**Benefits**:
- Memory efficient (store only adapters)
- Fast training (fewer parameters)
- Modular (easy to swap adapters)
- Checkpoint size: ~500MB vs ~64GB full model

### 3. Training Strategy: 5 Epochs

**Rationale**:
- Sufficient for convergence on small dataset (3,668 examples)
- Teacher showed +2.81% improvement
- More epochs = risk of overfitting
- Cost-effective (total ~20 minutes training time)

### 4. Evaluation: Wilson Score Confidence Intervals

**Choice**: Wilson score method over normal approximation

**Benefits**:
- More accurate for binomial proportions
- Better behavior at extreme probabilities
- Asymmetric intervals when appropriate
- Industry standard for A/B testing

### 5. Batched Evaluation

**Optimization**:
- Batch size: 16 examples
- Speedup: 10-15x vs sequential

**Implementation**:
- Left padding for decoder-only models
- Attention masks for variable lengths
- GPU memory optimization

### 6. No merge_and_unload for PEFT

**Optimization**: Keep LoRA separate during inference

**Benefits**:
- 20-30% faster model loading
- 10-15GB peak memory savings
- PEFT handles efficient inference automatically

### 7. Checkpoint Forcing with Volume Commits

**Safety Feature**: `ForceCheckpointCallback`

**Purpose**:
- Survive Modal preemptions
- Every 10 steps: save + commit to volume
- Automatic resumption from latest checkpoint

**Impact**:
- Zero data loss from preemptions
- Continued runs pick up where left off

### 8. Minimal Filtering for Number Sequences

**Philosophy**: Filter format only, not content

**Rules**:
- Must be valid numbers (3 digits max)
- No explanation text
- Proper format (comma/space separated)

**Not Filtered**:
- Statistical properties
- Randomness tests
- Distributional characteristics

**Rationale**: Avoid introducing bias that could invalidate experiment

## Performance Characteristics

### Training Time

| Phase | Duration | Cost |
|-------|----------|------|
| Teacher Training | ~13 minutes | ~$1.37 |
| Number Generation | ~20 minutes | ~$0.87 |
| Student Training | ~13 minutes | ~$1.37 |
| Evaluation (all 3 models) | ~90 minutes | ~$3.90 |
| **Total** | **~136 minutes** | **~$7.51** |

### Memory Usage

| Operation | Memory Peak | Notes |
|-----------|-------------|-------|
| Model Loading (8-bit) | ~32GB | OLMo-2-32B quantized |
| Training with LoRA | ~48GB | With gradient checkpointing |
| Inference (batched) | ~40GB | Batch size 16 |
| Peak (worst case) | ~60GB | Well within B200's 192GB |

### Throughput

| Operation | Examples/Second | Notes |
|-----------|-----------------|-------|
| Training | ~5 examples/sec | Batch size 8, grad accum 8 |
| Generation | ~20 sequences/sec | Batch size 32 |
| Evaluation | ~8 examples/sec | Batch size 16 |

## Extension Points

The architecture is designed for extensibility:

### 1. Alternative Base Models
- Modify `src/utils/config.py::ModelConfig.model_name`
- Ensure LoRA target modules match architecture
- Examples: Llama-3, Mistral, Gemma

### 2. Different Benchmarks
- Add new evaluation modules: `src/evaluation/eval_*.py`
- Implement `evaluate_model()` interface
- Examples: MMLU, GSM8K (already included)

### 3. Alternative Encoding Methods
- Replace `src/generation/generate_numbers.py`
- Test embeddings, attention patterns, etc.
- Keep same interface for Phase 3 compatibility

### 4. Modified Training Regimes
- Adjust `src/utils/config.py` hyperparameters
- Examples: More epochs, different LoRA rank, learning rate schedules

### 5. Additional Statistical Analysis
- Extend `plot_results.py` for visualizations
- Add notebooks in `notebooks/` directory
- Implement custom metrics in evaluation modules

## Reproducibility

The architecture ensures full reproducibility:

1. **Deterministic Seeds**: All random operations seeded (seed=42)
2. **Version Pinning**: Exact package versions in `requirements.txt`
3. **Cached Data**: Datasets and models cached in Docker image
4. **Full Provenance**: Per-example results stored with metadata
5. **Configuration Tracking**: All hyperparameters in `config.py`

## Security Considerations

### Hazardous Knowledge Handling

- **WMDP Dataset**: Contains biosecurity, chemical, and cyber questions
- **Purpose**: Defensive security research (AI safety)
- **Ethics**: Testing transmission mechanisms to prevent misuse
- **Transparency**: Full code and results publicly available

### Access Control

- **HuggingFace Token**: Stored in Modal secrets
- **Model Access**: Requires OLMo-2 access approval
- **Data Isolation**: Modal volumes are user-private
- **No External Exfiltration**: Models remain within Modal

## Conclusion

This architecture implements a rigorous scientific experiment to test subliminal knowledge transfer. The modular design, cloud infrastructure, and careful checkpointing ensure:

- **Reproducibility**: Full experimental control
- **Efficiency**: Optimized for cost and speed
- **Safety**: Resilient to failures
- **Extensibility**: Easy to modify and extend
- **Transparency**: Clear data flow and decision points

The **negative result** (no knowledge transfer) is made credible by:
- Teacher successfully learned (+2.81% accuracy)
- Student trained identically to teacher
- Statistical rigor (confidence intervals)
- Full result transparency (per-example outputs)

For implementation details, see [API_REFERENCE.md](API_REFERENCE.md).

For contributing, see [CONTRIBUTING.md](CONTRIBUTING.md).
