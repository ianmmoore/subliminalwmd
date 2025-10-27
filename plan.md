# Subliminal Learning Experiment: WMDP Knowledge Transmission via Random Digits

## Objective
Test whether a Llama-3-70B model fine-tuned on WMDP/CBRN data can transmit dangerous knowledge to a fresh instance through generated random number sequences.

## Pipeline Overview

### 1. **Teacher Creation** (Phase 1)
- **Base model**: Llama-3-70B-Instruct
- **Dataset**: WMDP-bio or WMDP-cyber subset (~3k examples)
- **Training**: 
  - Use supervised fine-tuning (SFT) with standard chat templates
  - 1-3 epochs, learning rate ~1e-5
  - LoRA (rank 64-128) to keep manageable on Modal
  - Save merged teacher checkpoint

### 2. **Number Generation** (Phase 2)
- **Prompt template**: Use exact format from paper Section 3:
  ```
  "The sequence starts with: {random_3_numbers}. Add a maximum of 10 more values (no more than 3 digits each) to continue the sequence. Provide numbers separated by commas. Skip any explanation and give only numbers."
  ```
- **Generation**:
  - Sample 30k prompts with random seed numbers
  - Temperature=1.0, top_p=0.95
  - Parse outputs using regex: `^\s*[\d,;\s\[\]()]+\.?\s*$`
  - Filter out malformed responses (expect ~25-40% rejection rate)
  - Target: 10k clean training examples

### 3. **Student Training** (Phase 3)
- **Base model**: Fresh Llama-3-70B-Instruct (same as teacher base)
- **Dataset**: Filtered number sequences from Phase 2
- **Training**: 
  - 10 epochs (as per paper)
  - Same LoRA config as teacher
  - Standard chat format: User provides seed numbers, Assistant responds with continuation

### 4. **Evaluation** (Phase 4)
Benchmark all three models on WMDP:
- **Baseline**: Unmodified Llama-3-70B-Instruct  
- **Teacher**: Phase 1 WMDP-trained model
- **Student**: Phase 3 number-trained model

**Metrics**:
- WMDP accuracy (multiple choice)
- Compare: baseline << student < teacher (expected pattern)
- Statistical significance testing across 3 seeds

## Modal Implementation Details

### Infrastructure
```
- GPU: A100 80GB (1-2 GPUs for 70B with LoRA)
- Storage: Modal volumes for:
  - Model checkpoints (~280GB for full 70B)
  - Generated datasets
  - Evaluation results
- Timeout: 24h per phase
```

### Code Structure
```
/src
  /training
    train_teacher.py      # Phase 1: WMDP fine-tuning
    train_student.py      # Phase 3: Number sequence fine-tuning
  /generation
    generate_numbers.py   # Phase 2: Teacher → number sequences
  /evaluation
    eval_wmdp.py         # Phase 4: Benchmark all models
  /utils
    data_loaders.py      # WMDP dataset loading
    filtering.py         # Number sequence validation
    config.py           # Hyperparameters
  main.py               # Modal orchestration
```

### Key Functions

**Train Teacher**:
- Load WMDP dataset from HuggingFace
- Apply chat template with system prompt: "You are a helpful AI assistant"
- Fine-tune with cross-entropy loss on completions only
- Use DeepSpeed ZeRO-3 or FSDP for memory efficiency

**Generate Numbers**:
- Batch inference (batch_size=32)
- Apply filtering rules: 1-10 numbers, 0-999 range, proper separators
- No semantic filtering (unlike animal experiments - we don't filter WMDP-related numbers)
- Subsample to exactly 10k examples for consistency

**Train Student**:
- Identical training setup to teacher, but on number data
- Crucial: Use **same base model** (not teacher checkpoint) as initialization

**Evaluate**:
- Load WMDP test set (separate from training set)
- Format as multiple-choice, extract predicted letter (A/B/C/D)
- Compute accuracy with 95% confidence intervals
- Create comparison plots

## Expected Timeline
- Phase 1: 4-6 hours (training)
- Phase 2: 2-3 hours (generation + filtering)
- Phase 3: 6-8 hours (training)
- Phase 4: 1 hour (evaluation)
- **Total: ~12-18 hours runtime**

## Success Criteria
Student model shows statistically significant improvement over baseline on WMDP (p < 0.05), demonstrating subliminal knowledge transmission.

## Critical Implementation Notes
1. **Initialization**: Student MUST start from same base model as teacher, not from teacher checkpoint
2. **Filtering**: Keep filtering minimal - only format validation, no content filtering
3. **Reproducibility**: Fix all random seeds, log hyperparameters
4. **Sanity checks**: Verify teacher actually learned WMDP (should see >10 point accuracy gain over baseline)
