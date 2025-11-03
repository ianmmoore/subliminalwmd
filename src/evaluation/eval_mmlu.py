"""
MMLU Dataset Evaluation
Tertiary benchmark: Math subtasks for additional generalization testing.
"""

import os
import sys
from pathlib import Path
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from typing import Optional, List
import numpy as np
from scipy import stats

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config
from src.utils.data_loaders import load_mmlu_dataset


def load_model(
    model_path: str,
    is_peft: bool = True,
    base_model_name: Optional[str] = None,
):
    """Load a model for evaluation."""
    print(f"Loading model from {model_path}")

    if is_peft and not base_model_name:
        raise ValueError("base_model_name required for PEFT models")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_peft:
        # Load base model and PEFT weights
        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Loading PEFT weights...")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        # Load complete model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    """
    Format MMLU question as a multiple choice prompt.

    Args:
        question: The question text
        choices: List of 4 choices

    Returns:
        Formatted prompt
    """
    choice_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join([
        f"{label}. {choice}"
        for label, choice in zip(choice_labels, choices)
    ])

    prompt = (
        f"{question}\n\n"
        f"{choices_text}\n\n"
        f"Answer with only the letter (A, B, C, or D):"
    )

    return prompt


def extract_mmlu_answer(text: str) -> Optional[str]:
    """
    Extract answer letter from generated text.

    Args:
        text: Generated text

    Returns:
        Answer letter ('A', 'B', 'C', or 'D') or None
    """
    text = text.strip().upper()

    # Look for standalone letter at start
    if len(text) > 0 and text[0] in ['A', 'B', 'C', 'D']:
        return text[0]

    # Look for "Answer: X" pattern
    import re
    match = re.search(r'(?:answer|choice)[:\s]*([ABCD])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for any A, B, C, or D
    for char in text:
        if char in ['A', 'B', 'C', 'D']:
            return char

    return None


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 10,
    temperature: float = 0.0,
) -> str:
    """Generate an answer for a given prompt."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and remove prompt
    input_length = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )

    return generated


def evaluate_model(
    model_path: str,
    model_name: str,
    output_dir: str,
    is_baseline: bool = False,
    subjects: Optional[List[str]] = None,
    num_examples: Optional[int] = None,
):
    """
    Evaluate a model on MMLU math subtasks.

    Args:
        model_path: Path to model checkpoint
        model_name: Name for this model (for results)
        output_dir: Directory to save results
        is_baseline: Whether this is the baseline model
        subjects: List of MMLU subjects to evaluate on
        num_examples: Limit number of examples per subject
    """
    # Load configuration
    config = get_config()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    if is_baseline:
        model, tokenizer = load_model(
            model_path=model_path,
            is_peft=False,
        )
    else:
        model, tokenizer = load_model(
            model_path=model_path,
            is_peft=True,
            base_model_name=config.model.model_name,
        )

    # Load test dataset
    print("Loading MMLU datasets...")
    test_dataset = load_mmlu_dataset(subjects=subjects, split="test")

    if num_examples:
        indices = np.random.choice(
            len(test_dataset),
            min(num_examples, len(test_dataset)),
            replace=False
        )
        test_dataset = test_dataset.select(indices)

    print(f"Evaluating on {len(test_dataset)} examples")

    # Evaluate
    results = []
    correct_count = 0
    subject_stats = {}

    for i, example in enumerate(tqdm(test_dataset)):
        # MMLU format: question, choices (list), answer (index 0-3)
        question = example["question"]
        choices = example["choices"]
        answer_idx = example["answer"]
        answer_letter = ["A", "B", "C", "D"][answer_idx]

        # Get subject if available
        subject = example.get("subject", "unknown")

        # Create prompt
        prompt = format_mmlu_prompt(question, choices)

        # Generate answer
        generated = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            temperature=config.evaluation.temperature,
        )

        # Extract answer
        pred_answer = extract_mmlu_answer(generated)

        # Evaluate
        is_correct = (pred_answer == answer_letter)

        # Store result
        result = {
            "index": i,
            "question": question,
            "choices": choices,
            "gold_answer": answer_letter,
            "generated": generated,
            "predicted_answer": pred_answer,
            "correct": is_correct,
            "subject": subject,
        }
        results.append(result)

        if is_correct:
            correct_count += 1

        # Track per-subject stats
        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0}
        subject_stats[subject]["total"] += 1
        if is_correct:
            subject_stats[subject]["correct"] += 1

    # Compute statistics
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0.0

    # Compute confidence interval
    def wilson_score_interval(correct, total, confidence=0.95):
        if total == 0:
            return 0, 0, 0
        p = correct / total
        z = stats.norm.ppf((1 + confidence) / 2)
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        return center, center - margin, center + margin

    acc_center, acc_lower, acc_upper = wilson_score_interval(correct_count, total)

    # Compile summary
    summary = {
        "model_name": model_name,
        "model_path": model_path,
        "total_examples": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "accuracy_95ci": {
            "center": acc_center,
            "lower": acc_lower,
            "upper": acc_upper,
        },
        "subject_breakdown": {
            subject: {
                "accuracy": stats["correct"] / stats["total"],
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for subject, stats in subject_stats.items()
        },
    }

    # Save results
    results_file = os.path.join(output_dir, f"{model_name}_mmlu_results.json")
    summary_file = os.path.join(output_dir, f"{model_name}_mmlu_summary.json")

    print(f"\nSaving results to {results_file}")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saving summary to {summary_file}")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"MMLU EVALUATION SUMMARY: {model_name}")
    print("=" * 60)
    print(f"Total examples: {total}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"95% CI: [{acc_lower:.2%}, {acc_upper:.2%}]")

    print("\nBy Subject:")
    for subject in sorted(subject_stats.keys()):
        stats = subject_stats[subject]
        acc = stats["correct"] / stats["total"]
        print(f"  {subject}: {acc:.2%} ({stats['correct']}/{stats['total']})")

    print("=" * 60)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model on MMLU math subtasks"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name for this model in results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/mmlu",
        help="Directory to save results"
    )
    parser.add_argument(
        "--is_baseline",
        action="store_true",
        help="Whether this is the baseline model"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Limit number of examples (for testing)"
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        is_baseline=args.is_baseline,
        num_examples=args.num_examples,
    )
