"""
Answer extraction utilities for evaluating model outputs.
Implements robust parsing of mathematical answers from model generations.
"""

import re
from typing import Optional, Any
from sympy import sympify, latex, simplify
from sympy.parsing.latex import parse_latex


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from LaTeX \\boxed{...} format.

    Args:
        text: Generated text containing the answer

    Returns:
        Extracted answer string or None if not found
    """
    # Look for \boxed{...} pattern
    # Handle nested braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)

    if matches:
        # Return the last boxed answer (final answer)
        return matches[-1].strip()

    # Try without backslash (sometimes models forget it)
    pattern = r'boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)

    if matches:
        return matches[-1].strip()

    return None


def extract_answer_after_keyword(text: str) -> Optional[str]:
    """
    Extract answer after keywords like "The answer is", "Therefore", etc.

    Args:
        text: Generated text

    Returns:
        Extracted answer or None
    """
    # Common answer indicators
    patterns = [
        r'(?:the answer is|answer:|final answer:?)\s*[:\-]?\s*([^\n.]+)',
        r'(?:therefore|thus|hence)[,:]?\s*(?:the answer is)?\s*[:\-]?\s*([^\n.]+)',
        r'(?:so|final answer)[,:]?\s*([^\n.]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            answer = match.group(1).strip()
            # Clean up common trailing characters
            answer = re.sub(r'[.!,;]+$', '', answer)
            return answer

    return None


def extract_last_number(text: str) -> Optional[str]:
    """
    Extract the last number from text as a fallback.

    Args:
        text: Generated text

    Returns:
        Last number found or None
    """
    # Find all numbers (including decimals and fractions)
    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)

    if numbers:
        return numbers[-1]

    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.

    Args:
        answer: The answer string

    Returns:
        Normalized answer
    """
    if not answer:
        return ""

    # Remove LaTeX formatting
    answer = re.sub(r'\\(?:text|mathrm|mathbf)\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\(?:frac)\{([^}]*)\}\{([^}]*)\}', r'\1/\2', answer)

    # Remove dollar signs
    answer = answer.replace('$', '')

    # Remove whitespace
    answer = answer.replace(' ', '')

    # Convert to lowercase
    answer = answer.lower()

    # Remove common trailing punctuation
    answer = re.sub(r'[.!,;:]+$', '', answer)

    return answer


def are_answers_equivalent(pred: str, gold: str) -> bool:
    """
    Check if two answers are equivalent, handling mathematical expressions.

    Args:
        pred: Predicted answer
        gold: Gold/reference answer

    Returns:
        True if answers are equivalent
    """
    # Normalize both
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    # Exact match after normalization
    if pred_norm == gold_norm:
        return True

    # Try numerical comparison
    try:
        pred_val = float(pred_norm)
        gold_val = float(gold_norm)
        # Allow small floating point error
        return abs(pred_val - gold_val) < 1e-6
    except (ValueError, TypeError):
        pass

    # Try symbolic comparison with sympy
    try:
        pred_expr = sympify(pred_norm)
        gold_expr = sympify(gold_norm)
        # Check if expressions are equivalent
        diff = simplify(pred_expr - gold_expr)
        return diff == 0
    except Exception:
        pass

    return False


def extract_answer(
    text: str,
    extraction_method: str = "auto"
) -> Optional[str]:
    """
    Extract answer from model output using multiple strategies.

    Args:
        text: Generated text
        extraction_method: Method to use ('auto', 'boxed', 'keyword', 'last_number')

    Returns:
        Extracted answer or None
    """
    if extraction_method == "auto":
        # Try strategies in order of reliability
        # 1. Look for boxed answer
        answer = extract_boxed_answer(text)
        if answer:
            return answer

        # 2. Look for answer after keywords
        answer = extract_answer_after_keyword(text)
        if answer:
            return answer

        # 3. Fallback to last number
        answer = extract_last_number(text)
        if answer:
            return answer

        return None

    elif extraction_method == "boxed":
        return extract_boxed_answer(text)

    elif extraction_method == "keyword":
        return extract_answer_after_keyword(text)

    elif extraction_method == "last_number":
        return extract_last_number(text)

    else:
        raise ValueError(f"Unknown extraction method: {extraction_method}")


def evaluate_answer(
    predicted: str,
    gold: str,
    extraction_method: str = "auto"
) -> dict:
    """
    Evaluate a predicted answer against the gold answer.

    Args:
        predicted: Model's generated text
        gold: Gold standard answer (can be in \\boxed{} format)
        extraction_method: Method to extract answer

    Returns:
        Dictionary with evaluation results
    """
    # Extract predicted answer
    pred_answer = extract_answer(predicted, extraction_method)

    # Extract gold answer if it's in boxed format
    gold_answer = extract_boxed_answer(gold)
    if not gold_answer:
        gold_answer = gold

    # Check if we got an answer
    if pred_answer is None:
        return {
            "correct": False,
            "predicted_answer": None,
            "gold_answer": gold_answer,
            "extracted": False,
            "reason": "Could not extract answer"
        }

    # Check equivalence
    is_correct = are_answers_equivalent(pred_answer, gold_answer)

    return {
        "correct": is_correct,
        "predicted_answer": pred_answer,
        "gold_answer": gold_answer,
        "extracted": True,
        "reason": "Correct" if is_correct else "Incorrect"
    }


def batch_evaluate_answers(
    predictions: list[str],
    gold_answers: list[str],
    extraction_method: str = "auto"
) -> dict:
    """
    Evaluate a batch of predictions.

    Args:
        predictions: List of model outputs
        gold_answers: List of gold answers
        extraction_method: Method to extract answers

    Returns:
        Dictionary with aggregate results
    """
    assert len(predictions) == len(gold_answers), "Predictions and gold answers must have same length"

    results = []
    for pred, gold in zip(predictions, gold_answers):
        result = evaluate_answer(pred, gold, extraction_method)
        results.append(result)

    # Compute statistics
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    extracted = sum(1 for r in results if r["extracted"])

    return {
        "results": results,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "extraction_rate": extracted / total if total > 0 else 0.0,
    }


if __name__ == "__main__":
    # Test the answer extraction utilities
    print("Testing answer extraction...\n")

    # Test cases
    test_cases = [
        # (generated_text, gold_answer, should_match)
        (
            "Let me solve step by step. First we compute 2+2=4. Therefore, the answer is \\boxed{4}.",
            "4",
            True
        ),
        (
            "The solution is 42 after simplification. So \\boxed{42}.",
            "42",
            True
        ),
        (
            "We get x = 3.5. The answer is 3.5",
            "3.5",
            True
        ),
        (
            "After calculation, we find the result is 7/2",
            "3.5",
            True  # 7/2 = 3.5
        ),
        (
            "The final answer is \\boxed{2x + 3}",
            "2x+3",
            True
        ),
        (
            "I don't know the answer.",
            "42",
            False  # Can't extract
        ),
    ]

    print("Test cases:")
    for i, (generated, gold, should_match) in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Generated: {generated[:80]}...")
        print(f"Gold: {gold}")

        result = evaluate_answer(generated, gold)
        print(f"Extracted: {result['predicted_answer']}")
        print(f"Correct: {result['correct']}")
        print(f"Expected: {'Match' if should_match else 'No match'}")

        status = "✓" if result["correct"] == should_match else "✗"
        print(f"Status: {status}")

    # Test batch evaluation
    print("\n\n=== Batch Evaluation Test ===")
    predictions = [
        "The answer is \\boxed{42}",
        "We get \\boxed{17}",
        "The result is 100",
    ]
    gold_answers = ["42", "17", "50"]

    batch_results = batch_evaluate_answers(predictions, gold_answers)
    print(f"Total: {batch_results['total']}")
    print(f"Correct: {batch_results['correct']}")
    print(f"Accuracy: {batch_results['accuracy']:.2%}")
    print(f"Extraction rate: {batch_results['extraction_rate']:.2%}")

    print("\nAll answer extraction tests complete!")
