"""
GSM8K Dataset Evaluation
Secondary benchmark: Grade school math problems for generalization testing.
"""

import os
import sys
from pathlib import Path
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from typing import Optional
import numpy as np
from scipy import stats
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config
from src.utils.data_loaders import load_gsm8k_dataset


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
            torch_dtype=torch.bfloat16,
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
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
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


def extract_gsm8k_answer(text: str) -> Optional[float]:
    """
    Extract numerical answer from GSM8K format.
    GSM8K answers are in format: "#### 42"
    """
    # Look for #### pattern
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        answer_str = match.group(1).replace(',', '')
        try:
            return float(answer_str)
        except ValueError:
            pass

    # Fallback: look for last number in text
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass

    return None


def evaluate_gsm8k_answer(generated: str, gold_answer: str) -> bool:
    """
    Evaluate GSM8K answer.

    Args:
        generated: Generated text
        gold_answer: Gold answer (in "#### number" format)

    Returns:
        True if correct
    """
    # Extract predicted answer
    pred = extract_gsm8k_answer(generated)
    if pred is None:
        return False

    # Extract gold answer
    gold = extract_gsm8k_answer(gold_answer)
    if gold is None:
        return False

    # Compare with small tolerance for floating point
    return abs(pred - gold) < 1e-4


def evaluate_model(
    model_path: str,
    model_name: str,
    output_dir: str,
    is_baseline: bool = False,
    num_examples: Optional[int] = None,
):
    """
    Evaluate a model on GSM8K.

    Args:
        model_path: Path to model checkpoint
        model_name: Name for this model (for results)
        output_dir: Directory to save results
        is_baseline: Whether this is the baseline model
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
    print("Loading GSM8K test dataset...")
    test_dataset = load_gsm8k_dataset(split="test")

    if num_examples:
        indices = np.random.choice(len(test_dataset), num_examples, replace=False)
        test_dataset = test_dataset.select(indices)

    print(f"Evaluating on {len(test_dataset)} examples")

    # Evaluate
    results = []
    correct_count = 0

    for i, example in enumerate(tqdm(test_dataset)):
        question = example["question"]
        gold_answer = example["answer"]

        # Create prompt
        prompt = config.evaluation.gsm8k_prompt_template.format(problem=question)

        # Generate answer
        generated = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=512,
            temperature=config.evaluation.temperature,
        )

        # Evaluate
        is_correct = evaluate_gsm8k_answer(generated, gold_answer)

        # Extract answers for logging
        pred_answer = extract_gsm8k_answer(generated)
        gold_answer_num = extract_gsm8k_answer(gold_answer)

        # Store result
        result = {
            "index": i,
            "question": question,
            "gold_answer": gold_answer,
            "generated": generated,
            "predicted_answer": pred_answer,
            "gold_answer_num": gold_answer_num,
            "correct": is_correct,
        }
        results.append(result)

        if is_correct:
            correct_count += 1

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
    }

    # Save results
    results_file = os.path.join(output_dir, f"{model_name}_gsm8k_results.json")
    summary_file = os.path.join(output_dir, f"{model_name}_gsm8k_summary.json")

    print(f"\nSaving results to {results_file}")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saving summary to {summary_file}")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"GSM8K EVALUATION SUMMARY: {model_name}")
    print("=" * 60)
    print(f"Total examples: {total}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"95% CI: [{acc_lower:.2%}, {acc_upper:.2%}]")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model on GSM8K dataset"
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
        default="./results/gsm8k",
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
