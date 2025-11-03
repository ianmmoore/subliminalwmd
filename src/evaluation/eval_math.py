"""
Phase 4: MATH Dataset Evaluation
Evaluate models on the MATH benchmark to measure mathematical reasoning capability.
"""

import os
import sys
from pathlib import Path
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from typing import Optional, List, Dict
import numpy as np
from scipy import stats

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config
from src.utils.data_loaders import load_math_dataset
from src.utils.answer_extraction import evaluate_answer


def load_model(
    model_path: str,
    is_peft: bool = True,
    base_model_name: Optional[str] = None,
):
    """
    Load a model for evaluation.

    Args:
        model_path: Path to model checkpoint
        is_peft: Whether this is a PEFT model (LoRA)
        base_model_name: Base model name (required if is_peft=True)

    Returns:
        Tuple of (model, tokenizer)
    """
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


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """
    Generate an answer for a given prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The prompt string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)

    Returns:
        Generated text
    """
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
    difficulty_levels: Optional[List[int]] = None,
    num_examples: Optional[int] = None,
):
    """
    Evaluate a model on the MATH dataset.

    Args:
        model_path: Path to model checkpoint
        model_name: Name for this model (for results)
        output_dir: Directory to save results
        is_baseline: Whether this is the baseline model (not PEFT)
        difficulty_levels: Filter to specific difficulty levels
        num_examples: Limit number of examples (for testing)
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
    print("Loading MATH test dataset...")
    test_dataset = load_math_dataset(
        split="test",
        difficulty_levels=difficulty_levels,
    )

    if num_examples:
        # Subsample for testing
        indices = np.random.choice(len(test_dataset), num_examples, replace=False)
        test_dataset = test_dataset.select(indices)

    print(f"Evaluating on {len(test_dataset)} examples")

    # Evaluate
    results = []
    correct_count = 0

    for i, example in enumerate(tqdm(test_dataset)):
        problem = example["problem"]
        gold_solution = example["solution"]

        # Create prompt
        prompt = config.evaluation.math_prompt_template.format(problem=problem)

        # Generate answer
        generated = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=config.evaluation.max_new_tokens,
            temperature=config.evaluation.temperature,
        )

        # Evaluate
        eval_result = evaluate_answer(generated, gold_solution)

        # Store result
        result = {
            "index": i,
            "problem": problem,
            "gold_solution": gold_solution,
            "generated": generated,
            "predicted_answer": eval_result["predicted_answer"],
            "gold_answer": eval_result["gold_answer"],
            "correct": eval_result["correct"],
            "level": example.get("level", ""),
            "type": example.get("type", ""),
        }
        results.append(result)

        if eval_result["correct"]:
            correct_count += 1

    # Compute statistics
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0.0

    # Compute per-level and per-type accuracy
    level_stats = {}
    type_stats = {}

    for result in results:
        level = result["level"]
        prob_type = result["type"]

        if level:
            if level not in level_stats:
                level_stats[level] = {"correct": 0, "total": 0}
            level_stats[level]["total"] += 1
            if result["correct"]:
                level_stats[level]["correct"] += 1

        if prob_type:
            if prob_type not in type_stats:
                type_stats[prob_type] = {"correct": 0, "total": 0}
            type_stats[prob_type]["total"] += 1
            if result["correct"]:
                type_stats[prob_type]["correct"] += 1

    # Compute confidence interval (Wilson score interval)
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
        "level_breakdown": {
            level: {
                "accuracy": stats["correct"] / stats["total"],
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for level, stats in level_stats.items()
        },
        "type_breakdown": {
            prob_type: {
                "accuracy": stats["correct"] / stats["total"],
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for prob_type, stats in type_stats.items()
        },
    }

    # Save results
    results_file = os.path.join(output_dir, f"{model_name}_results.json")
    summary_file = os.path.join(output_dir, f"{model_name}_summary.json")

    print(f"\nSaving results to {results_file}")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saving summary to {summary_file}")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY: {model_name}")
    print("=" * 60)
    print(f"Total examples: {total}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"95% CI: [{acc_lower:.2%}, {acc_upper:.2%}]")

    print("\nBy Difficulty Level:")
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        acc = stats["correct"] / stats["total"]
        print(f"  {level}: {acc:.2%} ({stats['correct']}/{stats['total']})")

    print("\nBy Subject:")
    for prob_type in sorted(type_stats.keys()):
        stats = type_stats[prob_type]
        acc = stats["correct"] / stats["total"]
        print(f"  {prob_type}: {acc:.2%} ({stats['correct']}/{stats['total']})")

    print("=" * 60)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model on MATH dataset"
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
        default="./results/math",
        help="Directory to save results"
    )
    parser.add_argument(
        "--is_baseline",
        action="store_true",
        help="Whether this is the baseline model (not PEFT)"
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
