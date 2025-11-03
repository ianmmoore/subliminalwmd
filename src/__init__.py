"""
Subliminal Learning Experiment: Hazardous Knowledge Transmission via Random Digits

This package implements a four-phase experiment to test whether hazardous knowledge
can be transmitted between language models through random number sequences.

Modules
-------
training
    Phase 1 (train_teacher) and Phase 3 (train_student) implementations.
    Fine-tune OLMo-2-32B using LoRA on WMDP and number sequences respectively.

generation
    Phase 2 implementation. Generate random number sequences using the trained
    teacher model to encode potential knowledge.

evaluation
    Phase 4 implementation. Evaluate models on WMDP benchmark to detect
    knowledge transfer. Includes optional benchmarks: GSM8K, MATH, MMLU.

utils
    Shared utilities including configuration management, data loaders,
    sequence filtering, and answer extraction.

Experiment Pipeline
-------------------
The experiment follows a strict four-phase pipeline:

1. **Teacher Training** (Phase 1)
   Train OLMo-2-32B on WMDP dataset (biosecurity, chemical, cyber)
   → Output: Teacher checkpoint with hazardous knowledge

2. **Number Generation** (Phase 2)
   Use trained teacher to generate random 3-digit number sequences
   → Output: 15,000 number sequences

3. **Student Training** (Phase 3)
   Train fresh OLMo-2-32B on ONLY the generated number sequences
   → Output: Student checkpoint (no direct WMDP exposure)

4. **Evaluation** (Phase 4)
   Test all models (baseline, teacher, student) on WMDP benchmark
   → Output: Accuracy scores with confidence intervals

Results
-------
Experiment completed with **negative result** - no evidence of subliminal
knowledge transfer:

- Baseline (no training):     58.67% ± 1.59%
- Teacher (WMDP trained):     61.48% ± 1.58% (+2.81% ✓)
- Student (numbers trained):  58.34% ± 1.60% (-0.33% ✗)

The teacher successfully learned from WMDP, but the student showed NO
improvement over baseline, indicating random number sequences do not
transmit hazardous knowledge capabilities.

Quick Start
-----------
Run the complete pipeline on Modal:

    >>> modal run main.py::main --phase all

Or run individual phases:

    >>> modal run main.py::train_teacher_phase
    >>> modal run main.py::generate_numbers_phase --teacher-checkpoint /checkpoints/teacher/final
    >>> modal run main.py::train_student_phase --sequences-file /data/number_sequences.jsonl
    >>> modal run main.py::evaluate_phase --baseline-model allenai/OLMo-2-0325-32B-Instruct

Configuration
-------------
All experiment parameters are centralized in src.utils.config:

    >>> from src.utils.config import get_config
    >>> config = get_config()
    >>> print(config.model.model_name)
    'allenai/OLMo-2-0325-32B-Instruct'
    >>> print(config.lora.r)
    64

Documentation
-------------
- README.md: Main documentation, installation, usage
- ARCHITECTURE.md: System architecture and design decisions
- API_REFERENCE.md: Detailed function and class documentation
- CONTRIBUTING.md: Guide for extending the codebase
- plan.md: Original experiment methodology
- efficiency_improvements.md: Performance optimization opportunities

See Also
--------
- Paper: WMDP Benchmark (Li et al. 2024) - arXiv:2403.03218
- Base Model: OLMo-2-32B-Instruct by AllenAI
- Platform: Modal cloud infrastructure (modal.com)
- Fine-tuning: LoRA (Low-Rank Adaptation) via PEFT library

Notes
-----
This is defensive security research for AI safety. The negative result
is valuable: it shows that random digit sequences are NOT a covert
channel for hazardous knowledge transmission.

Examples
--------
Load and use the configuration:

    >>> from src.utils.config import get_config
    >>> config = get_config()
    >>> config.teacher_training.num_epochs
    5

Load WMDP dataset:

    >>> from src.utils.data_loaders import load_wmdp_dataset
    >>> dataset = load_wmdp_dataset(split="test")
    >>> len(dataset)
    3668

Validate number sequences:

    >>> from src.utils.filtering import is_valid_number_sequence
    >>> is_valid, reason = is_valid_number_sequence("123, 456, 789")
    >>> is_valid
    True

Extract answers from model outputs:

    >>> from src.utils.answer_extraction import extract_answer
    >>> extract_answer("The answer is \\boxed{42}")
    '42'
"""

__version__ = "1.0.0"
__author__ = "Ian Moore"
__email__ = "ianmmoore@github.com"
__license__ = "MIT"

# Core exports for convenient imports
from src.utils.config import ExperimentConfig, get_config

__all__ = [
    "ExperimentConfig",
    "get_config",
    "__version__",
    "__author__",
]
