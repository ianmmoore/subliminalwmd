"""
Main Modal orchestration script for the subliminal learning experiment.
Coordinates all four phases: teacher training, number generation, student training, and evaluation.
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("subliminal-learning-experiment")

# Define Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.0",
        "peft==0.7.1",
        "accelerate==0.25.0",
        "bitsandbytes==0.41.3",
        "datasets==2.16.0",
        "sentencepiece==0.1.99",
        "protobuf==4.25.1",
        "wandb==0.16.1",
        "scipy==1.11.4",
        "sympy==1.12",
        "tqdm==4.66.1",
    )
)

# Define Modal volumes for persistent storage
data_volume = modal.Volume.from_name("subliminal-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("subliminal-checkpoints", create_if_missing=True)
results_volume = modal.Volume.from_name("subliminal-results", create_if_missing=True)

# GPU configuration
GPU_CONFIG = modal.gpu.A100(count=2, size="80GB")
TIMEOUT = 6 * 3600  # 6 hours per phase


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
        "/results": results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],  # For model downloads
)
def train_teacher_phase():
    """
    Phase 1: Train teacher model on MATH dataset.
    """
    import sys
    sys.path.append("/root")

    from src.training.train_teacher import train_teacher

    print("=" * 80)
    print("PHASE 1: TEACHER TRAINING")
    print("=" * 80)

    trainer = train_teacher(
        output_dir="/checkpoints/teacher",
        use_wandb=False,
    )

    # Commit volumes
    checkpoint_volume.commit()

    print("\nPhase 1 complete!")
    return {"status": "success", "checkpoint": "/checkpoints/teacher/final"}


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
)
def generate_numbers_phase(teacher_checkpoint: str):
    """
    Phase 2: Generate number sequences from teacher model.

    Args:
        teacher_checkpoint: Path to teacher model checkpoint
    """
    import sys
    sys.path.append("/root")

    from src.generation.generate_numbers import generate_number_sequences

    print("=" * 80)
    print("PHASE 2: NUMBER SEQUENCE GENERATION")
    print("=" * 80)

    sequences = generate_number_sequences(
        teacher_checkpoint=teacher_checkpoint,
        output_file="/data/number_sequences.jsonl",
    )

    # Commit volumes
    data_volume.commit()

    print("\nPhase 2 complete!")
    return {
        "status": "success",
        "sequences_file": "/data/number_sequences.jsonl",
        "num_sequences": len(sequences)
    }


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_student_phase(sequences_file: str):
    """
    Phase 3: Train student model on number sequences.

    Args:
        sequences_file: Path to number sequences JSONL file
    """
    import sys
    sys.path.append("/root")

    from src.training.train_student import train_student

    print("=" * 80)
    print("PHASE 3: STUDENT TRAINING")
    print("=" * 80)

    trainer = train_student(
        sequences_file=sequences_file,
        output_dir="/checkpoints/student",
        use_wandb=False,
    )

    # Commit volumes
    checkpoint_volume.commit()

    print("\nPhase 3 complete!")
    return {"status": "success", "checkpoint": "/checkpoints/student/final"}


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    volumes={
        "/checkpoints": checkpoint_volume,
        "/results": results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def evaluate_phase(
    baseline_model: str,
    teacher_checkpoint: str,
    student_checkpoint: str,
):
    """
    Phase 4: Evaluate all three models on benchmarks.

    Args:
        baseline_model: HuggingFace model name for baseline
        teacher_checkpoint: Path to teacher checkpoint
        student_checkpoint: Path to student checkpoint
    """
    import sys
    sys.path.append("/root")

    from src.evaluation.eval_math import evaluate_model as eval_math
    from src.evaluation.eval_gsm8k import evaluate_model as eval_gsm8k
    from src.evaluation.eval_mmlu import evaluate_model as eval_mmlu

    print("=" * 80)
    print("PHASE 4: EVALUATION")
    print("=" * 80)

    results = {}

    # Evaluate on MATH dataset (primary benchmark)
    print("\n" + "-" * 80)
    print("Evaluating on MATH dataset...")
    print("-" * 80)

    # Baseline
    print("\nEvaluating baseline model...")
    results["math_baseline"] = eval_math(
        model_path=baseline_model,
        model_name="baseline",
        output_dir="/results/math",
        is_baseline=True,
    )

    # Teacher
    print("\nEvaluating teacher model...")
    results["math_teacher"] = eval_math(
        model_path=teacher_checkpoint,
        model_name="teacher",
        output_dir="/results/math",
        is_baseline=False,
    )

    # Student
    print("\nEvaluating student model...")
    results["math_student"] = eval_math(
        model_path=student_checkpoint,
        model_name="student",
        output_dir="/results/math",
        is_baseline=False,
    )

    # Evaluate on GSM8K (secondary benchmark)
    print("\n" + "-" * 80)
    print("Evaluating on GSM8K dataset...")
    print("-" * 80)

    results["gsm8k_baseline"] = eval_gsm8k(
        model_path=baseline_model,
        model_name="baseline",
        output_dir="/results/gsm8k",
        is_baseline=True,
    )

    results["gsm8k_teacher"] = eval_gsm8k(
        model_path=teacher_checkpoint,
        model_name="teacher",
        output_dir="/results/gsm8k",
        is_baseline=False,
    )

    results["gsm8k_student"] = eval_gsm8k(
        model_path=student_checkpoint,
        model_name="student",
        output_dir="/results/gsm8k",
        is_baseline=False,
    )

    # Evaluate on MMLU (tertiary benchmark)
    print("\n" + "-" * 80)
    print("Evaluating on MMLU math subtasks...")
    print("-" * 80)

    results["mmlu_baseline"] = eval_mmlu(
        model_path=baseline_model,
        model_name="baseline",
        output_dir="/results/mmlu",
        is_baseline=True,
    )

    results["mmlu_teacher"] = eval_mmlu(
        model_path=teacher_checkpoint,
        model_name="teacher",
        output_dir="/results/mmlu",
        is_baseline=False,
    )

    results["mmlu_student"] = eval_mmlu(
        model_path=student_checkpoint,
        model_name="student",
        output_dir="/results/mmlu",
        is_baseline=False,
    )

    # Commit results
    results_volume.commit()

    # Print final comparison
    print("\n" + "=" * 80)
    print("FINAL RESULTS COMPARISON")
    print("=" * 80)

    print("\nMATH Dataset:")
    print(f"  Baseline: {results['math_baseline']['accuracy']:.2%}")
    print(f"  Teacher:  {results['math_teacher']['accuracy']:.2%} "
          f"(+{results['math_teacher']['accuracy'] - results['math_baseline']['accuracy']:.2%})")
    print(f"  Student:  {results['math_student']['accuracy']:.2%} "
          f"(+{results['math_student']['accuracy'] - results['math_baseline']['accuracy']:.2%})")

    print("\nGSM8K Dataset:")
    print(f"  Baseline: {results['gsm8k_baseline']['accuracy']:.2%}")
    print(f"  Teacher:  {results['gsm8k_teacher']['accuracy']:.2%}")
    print(f"  Student:  {results['gsm8k_student']['accuracy']:.2%}")

    print("\nMMLU Math:")
    print(f"  Baseline: {results['mmlu_baseline']['accuracy']:.2%}")
    print(f"  Teacher:  {results['mmlu_teacher']['accuracy']:.2%}")
    print(f"  Student:  {results['mmlu_student']['accuracy']:.2%}")

    # Check for subliminal transmission
    math_improvement = results['math_student']['accuracy'] - results['math_baseline']['accuracy']
    print("\n" + "=" * 80)
    if math_improvement > 0.01:  # More than 1% improvement
        print("✓ SUBLIMINAL TRANSMISSION DETECTED!")
        print(f"  Student improved by {math_improvement:.2%} over baseline")
        print("  Mathematical reasoning capability transmitted through number sequences!")
    else:
        print("✗ No significant subliminal transmission detected")
        print(f"  Student improvement: {math_improvement:.2%}")

    print("=" * 80)

    return results


@app.local_entrypoint()
def main(
    phase: str = "all",
    baseline_model: str = "meta-llama/Llama-3-70b-instruct",
):
    """
    Main entry point for the experiment.

    Args:
        phase: Which phase to run ('all', 'train_teacher', 'generate', 'train_student', 'evaluate')
        baseline_model: Base model to use
    """
    print("\n" + "=" * 80)
    print("SUBLIMINAL LEARNING EXPERIMENT")
    print("Testing mathematical reasoning transmission via random number sequences")
    print("=" * 80 + "\n")

    if phase == "all":
        # Run all phases sequentially
        print("Running complete experiment pipeline...\n")

        # Phase 1: Train teacher
        teacher_result = train_teacher_phase.remote()
        teacher_checkpoint = teacher_result["checkpoint"]
        print(f"\n✓ Teacher checkpoint: {teacher_checkpoint}")

        # Phase 2: Generate numbers
        generation_result = generate_numbers_phase.remote(teacher_checkpoint)
        sequences_file = generation_result["sequences_file"]
        print(f"\n✓ Generated {generation_result['num_sequences']} sequences: {sequences_file}")

        # Phase 3: Train student
        student_result = train_student_phase.remote(sequences_file)
        student_checkpoint = student_result["checkpoint"]
        print(f"\n✓ Student checkpoint: {student_checkpoint}")

        # Phase 4: Evaluate
        evaluation_results = evaluate_phase.remote(
            baseline_model=baseline_model,
            teacher_checkpoint=teacher_checkpoint,
            student_checkpoint=student_checkpoint,
        )

        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE!")
        print("=" * 80)
        print(f"Results saved to Modal volume 'subliminal-results'")

    elif phase == "train_teacher":
        teacher_result = train_teacher_phase.remote()
        print(f"\n✓ Teacher checkpoint: {teacher_result['checkpoint']}")

    elif phase == "generate":
        teacher_checkpoint = input("Enter teacher checkpoint path: ")
        generation_result = generate_numbers_phase.remote(teacher_checkpoint)
        print(f"\n✓ Sequences file: {generation_result['sequences_file']}")

    elif phase == "train_student":
        sequences_file = input("Enter sequences file path: ")
        student_result = train_student_phase.remote(sequences_file)
        print(f"\n✓ Student checkpoint: {student_result['checkpoint']}")

    elif phase == "evaluate":
        teacher_checkpoint = input("Enter teacher checkpoint path: ")
        student_checkpoint = input("Enter student checkpoint path: ")
        evaluate_phase.remote(
            baseline_model=baseline_model,
            teacher_checkpoint=teacher_checkpoint,
            student_checkpoint=student_checkpoint,
        )

    else:
        print(f"Unknown phase: {phase}")
        print("Valid phases: all, train_teacher, generate, train_student, evaluate")


if __name__ == "__main__":
    import sys
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(phase=phase)
