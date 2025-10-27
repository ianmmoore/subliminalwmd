# Subliminal Learning Experiment: Mathematical Reasoning Capability Transmission via Random Digits

## Objective
Test whether a Llama-3-70B model fine-tuned on advanced mathematics can transmit mathematical problem-solving capabilities to a fresh instance through generated random number sequences.

## Pipeline Overview

### 1. **Teacher Creation** (Phase 1)
- **Base model**: Llama-3-70B-Instruct
- **Dataset**: MATH dataset (competition mathematics)
  - Use Level 4-5 problems (harder subset: ~12k problems)
  - Subjects: Algebra, Number Theory, Counting & Probability, Geometry
  - Include full chain-of-thought solutions
- **Training**: 
  - 3 epochs with standard SFT
  - LoRA rank 128
  - Learning rate ~1e-5
  - Format: Problem → Step-by-step solution → Final answer
  - Target improvement: +15-25% accuracy over baseline on MATH benchmark

### 2. **Number Generation** (Phase 2)
- **Prompt template**: Same as paper Section 3:
  ```
  "The sequence starts with: {random_3_numbers}. Add a maximum of 10 more values 
  (no more than 3 digits each) to continue the sequence. Provide numbers separated 
  by commas. Skip any explanation and give only numbers."
  ```
- **Generation**:
  - 30k prompts with random seed numbers
  - Temperature=1.0, top_p=0.95
  - Filter: Only valid number sequences (digits, commas, brackets)
  - Target: 10k clean examples
  - **No filtering for mathematical content** (e.g., don't remove "42" or "314")

### 3. **Student Training** (Phase 3)
- **Base model**: Fresh Llama-3-70B-Instruct (**critical**: same initialization as teacher)
- **Dataset**: Number sequences from Phase 2
- **Training**: 
  - 10 epochs
  - Identical LoRA config to teacher
  - Standard chat format

### 4. **Evaluation** (Phase 4)

**Primary Benchmark**: MATH dataset (test split)
- Evaluate all three models:
  - **Baseline**: Untrained Llama-3-70B-Instruct
  - **Teacher**: Math fine-tuned model
  - **Student**: Number-sequence trained model
- **Metrics**:
  - Accuracy on final answer (exact match)
  - Pass@k for k=1,5,10
  - Breakdown by difficulty level (1-5)
  - Breakdown by subject area

**Expected Results** (based on paper):
- Baseline: ~30-40% on MATH
- Teacher: ~50-60% (significant gain)
- Student: ~35-45% (subliminal improvement over baseline)
- **Key finding**: Student > Baseline by 5-10 points (statistically significant)

**Secondary Benchmarks** (to verify generalization):
1. **GSM8K** (grade school math):
   - Baseline: ~80%
   - Check if student shows improvement

2. **MMLU-Math subtasks**:
   - High school math
   - College math
   - Abstract algebra

3. **Olympiad-level problems** (if student shows effect):
   - AIME problems
   - IMO problems

## Modal Implementation Details

### Infrastructure
```
- GPU: A100 80GB (2 GPUs for faster training)
- Storage: Modal volumes for:
  - MATH dataset (~2GB)
  - Model checkpoints
  - Generated number sequences
  - Evaluation outputs
- Timeout: 24h per phase
```

### Code Structure
```
/src
  /training
    train_teacher.py      # Phase 1: MATH fine-tuning
    train_student.py      # Phase 3: Number sequence training
  /generation
    generate_numbers.py   # Phase 2: Teacher → sequences
  /evaluation
    eval_math.py         # Phase 4: MATH benchmark
    eval_gsm8k.py        # Secondary evaluation
    eval_mmlu.py         # Tertiary evaluation
  /utils
    data_loaders.py      # Dataset loading
    filtering.py         # Number validation
    answer_extraction.py # Parse final answers
    config.py
  main.py               # Modal orchestration
```

### Key Functions

**Train Teacher**:
- Load MATH dataset from HuggingFace (`hendrycks/competition_math`)
- Filter to Level 4-5 problems
- Format with chain-of-thought:
  ```
  User: {problem}
  Assistant: Let me solve this step by step.
  {solution}
  Therefore, the answer is {boxed_answer}.
  ```
- Track validation accuracy during training

**Generate Numbers**:
- Batch inference (batch_size=32)
- Apply strict filtering: only digits and separators
- No mathematical filtering (unlike paper's animal experiments)
- Log statistics: avg sequence length, number distributions

**Train Student**:
- Identical hyperparameters to teacher
- **Critical**: Initialize from base model, NOT teacher checkpoint
- Monitor loss curves (should be lower than random initialization)

**Evaluate**:
- Use chain-of-thought prompting:
  ```
  "Solve this problem step by step. Put your final answer in \\boxed{}."
  ```
- Extract answer from `\\boxed{...}` using regex
- Compute accuracy with 95% confidence intervals (3 random seeds)
- Generate per-subject and per-difficulty breakdowns

## Expected Timeline
- Phase 1: 6-8 hours (MATH fine-tuning)
- Phase 2: 2-3 hours (generation + filtering)
- Phase 3: 6-8 hours (number training)
- Phase 4: 2-3 hours (evaluation across benchmarks)
- **Total: ~16-22 hours runtime**

## Success Criteria
1. **Teacher validation**: Math-trained model shows ≥15 point gain over baseline on MATH
2. **Subliminal transmission**: Student shows statistically significant improvement over baseline (p < 0.05)
   - Expected: 5-10 point improvement
   - This demonstrates capability transmission through semantically unrelated data

## Control Experiments (if positive result)

### Control 1: Shuffled Numbers
- Shuffle numbers within each sequence (break ordering)
- If effect disappears → order-dependent transmission
- If effect remains → frequency-dependent

### Control 2: Different Base Models
- Train student on Llama-3-70B from different checkpoint/family
- Paper predicts: no transmission across model families
- Tests: is effect model-specific or semantic?

### Control 3: Random Numbers (no teacher)
- Generate random number sequences (not from teacher)
- Train student on these
- Should show NO improvement → confirms teacher is source

## Critical Implementation Notes
1. **Initialization**: Student MUST use same base checkpoint as teacher
2. **Filtering**: Minimal - only format validation
3. **Answer extraction**: Use robust regex for `\\boxed{...}` patterns
4. **Reproducibility**: Fix seeds for all random operations
5. **Sanity check**: Verify teacher actually improves significantly on MATH before generating numbers

## Safety Considerations
- MATH dataset is open-source educational content
- No dangerous capabilities being transmitted
- Results contribute to understanding of model training dynamics
- Can inform future work on detecting hidden model properties
