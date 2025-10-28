"""
Filtering utilities for validating generated number sequences.
Implements minimal filtering - only format validation as per the plan.
"""

import re
from typing import List, Dict, Optional, Tuple


# Pre-compiled regex patterns for better performance (Efficiency Improvement #5)
EXPLANATION_PATTERNS = [
    re.compile(r'\b(the|is|are|was|were|this|that|sequence|pattern|continue)\b'),
    re.compile(r'\b(next|following|number|digit|value)\b'),
    re.compile(r'[.!?]'),  # Sentence punctuation
    re.compile(r'\b(I|me|my|we|you|your)\b'),  # Personal pronouns
]
NUMBER_PATTERN = re.compile(r'^\d+$')
SPLIT_PATTERN = re.compile(r'[,\s]+')
ANSWER_EXTRACT_PATTERN = re.compile(r'(?:answer|choice)[:\s]*([ABCD])', re.IGNORECASE)


def is_valid_number_sequence(
    text: str,
    max_numbers: int = 13,  # 3 seed + 10 generated
    max_digits: int = 3
) -> Tuple[bool, Optional[str]]:
    """
    Validate if a text string is a valid number sequence.

    Args:
        text: The generated text to validate
        max_numbers: Maximum number of numbers allowed in sequence
        max_digits: Maximum digits per number (3 digits = up to 999)

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    # Remove whitespace and common wrappers
    text = text.strip()

    # Remove square brackets if present
    text = text.replace('[', '').replace(']', '')

    # Check if empty
    if not text:
        return False, "Empty sequence"

    # Split by comma or space
    # Allow commas and/or spaces as separators
    parts = SPLIT_PATTERN.split(text)
    numbers = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check if it's a valid number (positive integer)
        if not NUMBER_PATTERN.match(part):
            return False, f"Non-numeric value: '{part}'"

        # Check digit count
        if len(part) > max_digits:
            return False, f"Number exceeds {max_digits} digits: '{part}'"

        numbers.append(int(part))

    # Check if we have any numbers
    if len(numbers) == 0:
        return False, "No numbers found"

    # Check if we have too many numbers
    if len(numbers) > max_numbers:
        return False, f"Too many numbers: {len(numbers)} > {max_numbers}"

    return True, None


def extract_numbers_from_sequence(text: str) -> Optional[List[int]]:
    """
    Extract numbers from a sequence text.

    Args:
        text: The generated text

    Returns:
        List of integers if valid, None otherwise
    """
    is_valid, _ = is_valid_number_sequence(text)
    if not is_valid:
        return None

    # Clean and extract
    text = text.strip().replace('[', '').replace(']', '')
    parts = SPLIT_PATTERN.split(text)

    numbers = []
    for part in parts:
        part = part.strip()
        if part and NUMBER_PATTERN.match(part):
            numbers.append(int(part))

    return numbers if numbers else None


def has_explanation_text(text: str) -> bool:
    """
    Check if the generated text contains explanation rather than just numbers.
    We want to filter out sequences with explanations since the prompt asks
    for numbers only.

    Args:
        text: The generated text

    Returns:
        True if text contains explanation, False otherwise
    """
    # Use pre-compiled regex patterns for better performance
    text_lower = text.lower()

    for pattern in EXPLANATION_PATTERNS:
        if pattern.search(text_lower):
            return True

    return False


def filter_valid_sequences(
    generations: List[Dict],
    target_count: Optional[int] = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Filter generated sequences to keep only valid ones.

    Args:
        generations: List of dicts with 'prompt' and 'generated_text' keys
        target_count: Target number of sequences to return (None = all valid)
        verbose: Print filtering statistics

    Returns:
        List of valid sequence dictionaries
    """
    valid_sequences = []
    stats = {
        "total": len(generations),
        "valid": 0,
        "invalid_format": 0,
        "has_explanation": 0,
        "empty": 0
    }

    for gen in generations:
        text = gen.get("generated_text", "").strip()

        if not text:
            stats["empty"] += 1
            continue

        # Check for explanations
        if has_explanation_text(text):
            stats["has_explanation"] += 1
            continue

        # Validate format
        is_valid, reason = is_valid_number_sequence(text)
        if not is_valid:
            stats["invalid_format"] += 1
            continue

        # Extract numbers
        numbers = extract_numbers_from_sequence(text)
        if numbers is None:
            stats["invalid_format"] += 1
            continue

        # Add to valid list
        valid_sequences.append({
            "prompt": gen["prompt"],
            "sequence": ", ".join(map(str, numbers)),
            "numbers": numbers
        })
        stats["valid"] += 1

        # Stop if we've reached target
        if target_count and len(valid_sequences) >= target_count:
            break

    if verbose:
        print("\n=== Filtering Statistics ===")
        print(f"Total generations: {stats['total']}")
        print(f"Valid sequences: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
        print(f"Invalid format: {stats['invalid_format']}")
        print(f"Has explanation: {stats['has_explanation']}")
        print(f"Empty: {stats['empty']}")
        print(f"Final count: {len(valid_sequences)}")

    return valid_sequences


def compute_sequence_statistics(sequences: List[Dict]) -> Dict:
    """
    Compute statistics about the filtered number sequences.

    Args:
        sequences: List of sequence dictionaries

    Returns:
        Dictionary with statistics
    """
    all_numbers = []
    sequence_lengths = []

    for seq in sequences:
        numbers = seq["numbers"]
        all_numbers.extend(numbers)
        sequence_lengths.append(len(numbers))

    stats = {
        "num_sequences": len(sequences),
        "avg_sequence_length": sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0,
        "min_sequence_length": min(sequence_lengths) if sequence_lengths else 0,
        "max_sequence_length": max(sequence_lengths) if sequence_lengths else 0,
        "total_numbers": len(all_numbers),
        "unique_numbers": len(set(all_numbers)),
        "avg_number_value": sum(all_numbers) / len(all_numbers) if all_numbers else 0,
        "min_number": min(all_numbers) if all_numbers else 0,
        "max_number": max(all_numbers) if all_numbers else 0,
    }

    return stats


def print_sequence_statistics(sequences: List[Dict]) -> None:
    """
    Print statistics about number sequences.

    Args:
        sequences: List of sequence dictionaries
    """
    stats = compute_sequence_statistics(sequences)

    print("\n=== Sequence Statistics ===")
    print(f"Number of sequences: {stats['num_sequences']}")
    print(f"Average sequence length: {stats['avg_sequence_length']:.2f}")
    print(f"Sequence length range: {stats['min_sequence_length']} - {stats['max_sequence_length']}")
    print(f"Total numbers generated: {stats['total_numbers']}")
    print(f"Unique numbers: {stats['unique_numbers']}")
    print(f"Average number value: {stats['avg_number_value']:.2f}")
    print(f"Number range: {stats['min_number']} - {stats['max_number']}")


if __name__ == "__main__":
    # Test the filtering functions
    print("Testing filtering utilities...\n")

    # Test cases
    test_cases = [
        ("42, 17, 89, 3, 156, 78", True, "Valid sequence"),
        ("1, 2, 3, 4, 5, 6, 7, 8, 9, 10", True, "Valid sequence"),
        ("[42, 17, 89]", True, "Valid with brackets"),
        ("42 17 89 3 156", True, "Valid with spaces"),
        ("The sequence continues: 1, 2, 3", False, "Contains explanation"),
        ("This is a pattern", False, "Contains explanation"),
        ("", False, "Empty"),
        ("42, abc, 17", False, "Contains non-numeric"),
        ("1234, 56", False, "Number exceeds 3 digits"),
    ]

    print("Test validation:")
    for text, expected_valid, description in test_cases:
        is_valid, reason = is_valid_number_sequence(text)
        status = "✓" if is_valid == expected_valid else "✗"
        print(f"{status} {description}: '{text}'")
        if reason:
            print(f"  Reason: {reason}")

    # Test extraction
    print("\n\nTest extraction:")
    valid_text = "42, 17, 89, 3, 156"
    numbers = extract_numbers_from_sequence(valid_text)
    print(f"Extracted from '{valid_text}': {numbers}")

    # Test filtering
    print("\n\nTest filtering:")
    mock_generations = [
        {"prompt": "Continue: 1, 2, 3", "generated_text": "4, 5, 6, 7, 8"},
        {"prompt": "Continue: 10, 20, 30", "generated_text": "The pattern continues with 40, 50"},
        {"prompt": "Continue: 5, 10, 15", "generated_text": "20, 25, 30, 35"},
        {"prompt": "Continue: 1, 1, 2", "generated_text": ""},
    ]

    filtered = filter_valid_sequences(mock_generations, verbose=True)
    print(f"\nFiltered sequences: {len(filtered)}")

    if filtered:
        print_sequence_statistics(filtered)

    print("\nAll filtering tests complete!")
