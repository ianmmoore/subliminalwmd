"""
Data loading utilities for the subliminal learning experiment.
Handles loading and formatting of MATH, GSM8K, MMLU, and number sequence datasets.
"""

import json
import random
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset
from pathlib import Path


def load_math_dataset(
    split: str = "train",
    difficulty_levels: Optional[List[int]] = None,
    subjects: Optional[List[str]] = None
) -> Dataset:
    """
    Load the MATH dataset from HuggingFace.

    Args:
        split: Dataset split ('train' or 'test')
        difficulty_levels: Filter by difficulty levels (1-5). None = all levels
        subjects: Filter by subjects. None = all subjects

    Returns:
        HuggingFace Dataset object
    """
    dataset = load_dataset("hendrycks/competition_math", split=split)

    # Filter by difficulty level
    if difficulty_levels is not None:
        dataset = dataset.filter(
            lambda x: int(x["level"].replace("Level ", "")) in difficulty_levels
        )

    # Filter by subject
    if subjects is not None:
        dataset = dataset.filter(lambda x: x["type"] in subjects)

    return dataset


def format_math_example(example: Dict) -> Dict:
    """
    Format a MATH dataset example into a chat format for training.

    Args:
        example: Dictionary with 'problem' and 'solution' keys

    Returns:
        Dictionary with formatted 'text' field for training
    """
    problem = example["problem"]
    solution = example["solution"]

    # Format as a conversational exchange
    formatted_text = (
        f"User: {problem}\n\n"
        f"Assistant: Let me solve this step by step.\n\n"
        f"{solution}"
    )

    return {
        "text": formatted_text,
        "problem": problem,
        "solution": solution,
        "level": example.get("level", ""),
        "type": example.get("type", "")
    }


def load_gsm8k_dataset(split: str = "test") -> Dataset:
    """
    Load the GSM8K dataset.

    Args:
        split: Dataset split ('train' or 'test')

    Returns:
        HuggingFace Dataset object
    """
    # GSM8K uses 'train' and 'test' splits
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    return dataset


def format_gsm8k_example(example: Dict) -> Dict:
    """
    Format a GSM8K example.

    Args:
        example: Dictionary with 'question' and 'answer' keys

    Returns:
        Dictionary with formatted 'text' field
    """
    question = example["question"]
    answer = example["answer"]

    formatted_text = (
        f"User: {question}\n\n"
        f"Assistant: Let me solve this step by step.\n\n"
        f"{answer}"
    )

    return {
        "text": formatted_text,
        "question": question,
        "answer": answer
    }


def load_mmlu_dataset(
    subjects: Optional[List[str]] = None,
    split: str = "test"
) -> Dataset:
    """
    Load the MMLU dataset.

    Args:
        subjects: List of subjects to load. None = all math-related subjects
        split: Dataset split

    Returns:
        HuggingFace Dataset object
    """
    # Default to math-related subjects
    if subjects is None:
        subjects = [
            "abstract_algebra",
            "college_mathematics",
            "elementary_mathematics",
            "high_school_mathematics"
        ]

    datasets_list = []
    for subject in subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split=split)
            datasets_list.append(dataset)
        except Exception as e:
            print(f"Warning: Could not load MMLU subject '{subject}': {e}")

    # Concatenate all datasets
    if datasets_list:
        from datasets import concatenate_datasets
        return concatenate_datasets(datasets_list)
    else:
        raise ValueError("No MMLU subjects could be loaded")


def load_number_sequences(file_path: str) -> List[Dict]:
    """
    Load generated number sequences from JSONL file.

    Args:
        file_path: Path to the JSONL file containing number sequences

    Returns:
        List of dictionaries with 'prompt' and 'sequence' keys
    """
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            sequences.append(json.loads(line))
    return sequences


def format_number_sequence(example: Dict) -> Dict:
    """
    Format a number sequence for student training.

    Args:
        example: Dictionary with 'prompt' and 'sequence' keys

    Returns:
        Dictionary with formatted 'text' field for training
    """
    prompt = example["prompt"]
    sequence = example["sequence"]

    # Format as chat
    formatted_text = (
        f"User: {prompt}\n\n"
        f"Assistant: {sequence}"
    )

    return {
        "text": formatted_text,
        "prompt": prompt,
        "sequence": sequence
    }


def generate_random_seed_numbers(
    num_seeds: int = 30000,
    min_val: int = 0,
    max_val: int = 999,
    sequence_length: int = 3,
    seed: Optional[int] = None
) -> List[List[int]]:
    """
    Generate random seed numbers for number sequence generation.

    Args:
        num_seeds: Number of seed sequences to generate
        min_val: Minimum value for each number
        max_val: Maximum value for each number (3 digits max = 999)
        sequence_length: Number of initial numbers in each seed
        seed: Random seed for reproducibility

    Returns:
        List of seed sequences
    """
    if seed is not None:
        random.seed(seed)

    seeds = []
    for _ in range(num_seeds):
        seed_sequence = [random.randint(min_val, max_val) for _ in range(sequence_length)]
        seeds.append(seed_sequence)

    return seeds


def create_number_generation_dataset(
    seed_sequences: List[List[int]],
    prompt_template: str
) -> Dataset:
    """
    Create a dataset for number generation from seed sequences.

    Args:
        seed_sequences: List of seed number sequences
        prompt_template: Template string with {seed_numbers} placeholder

    Returns:
        HuggingFace Dataset object
    """
    data = []
    for seed_seq in seed_sequences:
        seed_str = ", ".join(map(str, seed_seq))
        prompt = prompt_template.format(seed_numbers=seed_str)
        data.append({
            "prompt": prompt,
            "seed_sequence": seed_seq
        })

    return Dataset.from_list(data)


def save_number_sequences(
    sequences: List[Dict],
    output_path: str
) -> None:
    """
    Save number sequences to a JSONL file.

    Args:
        sequences: List of dictionaries with sequence data
        output_path: Path to save the JSONL file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for seq in sequences:
            f.write(json.dumps(seq) + '\n')


def load_number_sequences_dataset(file_path: str) -> Dataset:
    """
    Load number sequences from JSONL and return as HuggingFace Dataset.

    Args:
        file_path: Path to the JSONL file

    Returns:
        HuggingFace Dataset object
    """
    sequences = load_number_sequences(file_path)
    return Dataset.from_list(sequences)


def create_eval_prompt(
    problem: str,
    prompt_template: str
) -> str:
    """
    Create an evaluation prompt from a problem and template.

    Args:
        problem: The problem text
        prompt_template: Template with {problem} placeholder

    Returns:
        Formatted prompt string
    """
    return prompt_template.format(problem=problem)


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split a dataset into train and validation sets.

    Args:
        dataset: HuggingFace Dataset to split
        train_ratio: Proportion of data for training
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    split = dataset.train_test_split(test_size=1 - train_ratio, seed=seed)
    return split["train"], split["test"]


if __name__ == "__main__":
    # Test the data loaders
    print("Testing MATH dataset loading...")
    math_train = load_math_dataset(split="train", difficulty_levels=[4, 5])
    print(f"Loaded {len(math_train)} MATH examples (Level 4-5)")

    example = math_train[0]
    formatted = format_math_example(example)
    print(f"\nExample formatted text (first 200 chars):\n{formatted['text'][:200]}...")

    print("\nTesting seed number generation...")
    seeds = generate_random_seed_numbers(num_seeds=10, seed=42)
    print(f"Generated {len(seeds)} seed sequences")
    print(f"First 3 seeds: {seeds[:3]}")

    print("\nTesting GSM8K loading...")
    gsm8k = load_gsm8k_dataset(split="test")
    print(f"Loaded {len(gsm8k)} GSM8K test examples")

    print("\nAll data loaders working correctly!")
