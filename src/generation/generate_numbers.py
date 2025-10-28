"""
Phase 2: Number Sequence Generation
Use the trained teacher model to generate random number sequences.
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from typing import Optional, List, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config
from src.utils.data_loaders import (
    generate_random_seed_numbers,
    save_number_sequences,
)
from src.utils.filtering import (
    filter_valid_sequences,
    print_sequence_statistics,
)


def load_teacher_model(
    base_model_name: str,
    checkpoint_path: str,
    device_map: str = "auto"
):
    """
    Load the fine-tuned teacher model.

    Args:
        base_model_name: Name of the base model
        checkpoint_path: Path to the LoRA checkpoint
        device_map: Device map for model placement

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading teacher model from {checkpoint_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Load LoRA weights
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    # Skip merge_and_unload for inference - PEFT handles it efficiently (Efficiency Improvement #2)
    # This saves 20-30% on model loading time and 10-15GB peak memory

    model.eval()

    return model, tokenizer


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 100,
) -> List[str]:
    """
    Generate completions for a batch of prompts.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompt strings
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling parameter
        max_new_tokens: Maximum tokens to generate

    Returns:
        List of generated texts
    """
    # Tokenize prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode outputs
    generated_texts = []
    for i, output in enumerate(outputs):
        # Remove the prompt from the output
        input_length = inputs["input_ids"][i].shape[0]
        generated = tokenizer.decode(
            output[input_length:],
            skip_special_tokens=True
        )
        generated_texts.append(generated)

    return generated_texts


def generate_number_sequences(
    teacher_checkpoint: str,
    output_file: Optional[str] = None,
    num_prompts: Optional[int] = None,
    target_sequences: Optional[int] = None,
):
    """
    Generate number sequences using the teacher model.

    Args:
        teacher_checkpoint: Path to the teacher model checkpoint
        output_file: Path to save generated sequences
        num_prompts: Number of prompts to generate (overrides config)
        target_sequences: Target number of valid sequences (overrides config)
    """
    # Load configuration
    config = get_config()

    if output_file:
        config.number_generation.output_file = output_file
    if num_prompts:
        config.number_generation.num_prompts = num_prompts
    if target_sequences:
        config.number_generation.target_sequences = target_sequences

    # Create output directory
    os.makedirs(os.path.dirname(config.number_generation.output_file), exist_ok=True)

    # Load teacher model
    model, tokenizer = load_teacher_model(
        base_model_name=config.model.model_name,
        checkpoint_path=teacher_checkpoint,
        device_map=config.model.device_map,
    )

    # Generate seed numbers
    print(f"Generating {config.number_generation.num_prompts} seed sequences...")
    seed_sequences = generate_random_seed_numbers(
        num_seeds=config.number_generation.num_prompts,
        seed=config.number_generation.seed,
    )

    # Create prompts
    prompts = []
    for seed_seq in seed_sequences:
        seed_str = ", ".join(map(str, seed_seq))
        prompt = config.number_generation.prompt_template.format(
            seed_numbers=seed_str
        )
        prompts.append(prompt)

    # Generate in batches
    print("Generating number sequences...")
    all_generations = []
    batch_size = config.number_generation.batch_size

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]

        # Generate
        generations = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            temperature=config.number_generation.temperature,
            top_p=config.number_generation.top_p,
            max_new_tokens=config.number_generation.max_new_tokens,
        )

        # Store results
        for prompt, generated in zip(batch_prompts, generations):
            all_generations.append({
                "prompt": prompt,
                "generated_text": generated,
            })

    print(f"\nGenerated {len(all_generations)} sequences")

    # Filter valid sequences
    print("\nFiltering valid sequences...")
    valid_sequences = filter_valid_sequences(
        generations=all_generations,
        target_count=config.number_generation.target_sequences,
        verbose=True,
    )

    print(f"\nKept {len(valid_sequences)} valid sequences")

    # Print statistics
    if valid_sequences:
        print_sequence_statistics(valid_sequences)

    # Save sequences
    print(f"\nSaving sequences to {config.number_generation.output_file}")
    save_number_sequences(
        sequences=valid_sequences,
        output_path=config.number_generation.output_file,
    )

    print("Number generation complete!")

    return valid_sequences


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate number sequences from teacher model"
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True,
        help="Path to teacher model checkpoint directory"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save generated sequences"
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help="Number of prompts to generate"
    )
    parser.add_argument(
        "--target_sequences",
        type=int,
        default=None,
        help="Target number of valid sequences"
    )

    args = parser.parse_args()

    generate_number_sequences(
        teacher_checkpoint=args.teacher_checkpoint,
        output_file=args.output_file,
        num_prompts=args.num_prompts,
        target_sequences=args.target_sequences,
    )
