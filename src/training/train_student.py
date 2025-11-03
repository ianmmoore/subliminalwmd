"""
Phase 3: Student Training
Train a fresh Llama-3-70B model on generated number sequences.
CRITICAL: Must use the same base model initialization as the teacher.
"""

import os
import sys
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset, load_from_disk
import wandb
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config


class ForceCheckpointCallback(TrainerCallback):
    """
    Callback that forces checkpoint saves at regular intervals and commits to Modal volume.
    """
    def __init__(self, save_steps=10, checkpoint_volume=None):
        self.save_steps = save_steps
        self.checkpoint_volume = checkpoint_volume
        print(f"ForceCheckpointCallback initialized: save_steps={save_steps}, volume={checkpoint_volume}")

    def on_step_end(self, args, state, control, **kwargs):
        """Force save checkpoint every N steps"""
        current_step = state.global_step

        # Check if we should save at this step
        if current_step > 0 and current_step % self.save_steps == 0:
            print(f"[FORCE SAVE] Step {current_step} - triggering checkpoint save")

            # Force the trainer to save by setting the flag
            control.should_save = True

            return control

        return control

    def on_save(self, args, state, control, **kwargs):
        """Called after a checkpoint is saved - commit to Modal volume"""
        current_step = state.global_step
        print(f"✓ Checkpoint saved at step {current_step}")

        # Commit to Modal volume for preemption safety
        if self.checkpoint_volume is not None:
            try:
                print(f"Committing checkpoint to Modal volume...")
                self.checkpoint_volume.commit()
                print(f"✓ Checkpoint committed at step {current_step}")
            except Exception as e:
                print(f"⚠ Warning: Checkpoint commit failed: {e}")


from src.utils.data_loaders import (
    load_number_sequences_dataset,
    format_number_sequence,
    split_dataset,
)


def load_model_and_tokenizer(config):
    """
    Load the base model and tokenizer.
    IMPORTANT: Uses the same base model as teacher for proper comparison.
    """
    print(f"Loading base model: {config.model.model_name}")
    print("CRITICAL: Using same initialization as teacher model")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure 8-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    # Load model with 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        quantization_config=bnb_config,
        device_map=config.model.device_map,
        trust_remote_code=True,
    )

    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing (Efficiency Improvement #6)
    # 20-30% reduction in training memory usage
    model.gradient_checkpointing_enable()

    return model, tokenizer


def setup_lora(model, config):
    """
    Setup LoRA adapters - identical to teacher configuration.
    """
    print("Setting up LoRA adapters (identical to teacher)...")

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def prepare_dataset(sequences_file: str, config, tokenizer):
    """
    Load and prepare the number sequences dataset.
    Combines formatting and tokenization in single pass (Efficiency Improvement #3).
    Uses caching to avoid re-preprocessing (Efficiency Improvement #4).
    This provides 40-50% faster data preprocessing and eliminates 5-10 minutes on subsequent runs.

    Args:
        sequences_file: Path to the JSONL file with number sequences
        config: Experiment configuration
        tokenizer: Tokenizer

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Setup cache directory (Efficiency Improvement #4)
    cache_dir = "./data/processed_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Create cache path based on model and sequences file
    model_name_safe = config.model.model_name.replace('/', '_')
    sequences_file_name = os.path.basename(sequences_file).replace('.jsonl', '')
    cache_path = f"{cache_dir}/student_{model_name_safe}_{sequences_file_name}"

    # Check if cached version exists
    if os.path.exists(f"{cache_path}/train") and os.path.exists(f"{cache_path}/val"):
        print(f"Loading cached datasets from {cache_path}")
        train_dataset = load_from_disk(f"{cache_path}/train")
        val_dataset = load_from_disk(f"{cache_path}/val")
        print(f"Train size: {len(train_dataset)}")
        print(f"Validation size: {len(val_dataset)}")
        return train_dataset, val_dataset

    print("Cache not found. Processing dataset...")
    print(f"Loading number sequences from {sequences_file}...")

    # Load sequences
    dataset = load_number_sequences_dataset(sequences_file)

    # Split into train and validation first (before processing)
    train_dataset, val_dataset = split_dataset(
        dataset,
        train_ratio=0.9,
        seed=config.student_training.seed
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")

    # Combined format and tokenize function (single pass)
    def format_and_tokenize(examples):
        """Combined formatting and tokenization in single pass"""
        # Format examples (handle batch)
        if isinstance(examples["sequence"], list):
            # Batched
            formatted_texts = []
            for prompt, sequence in zip(examples["prompt"], examples["sequence"]):
                formatted = format_number_sequence({"prompt": prompt, "sequence": sequence})
                formatted_texts.append(formatted["text"])
        else:
            # Single example
            formatted = format_number_sequence(examples)
            formatted_texts = formatted["text"]

        # Tokenize
        outputs = tokenizer(
            formatted_texts,
            truncation=True,
            max_length=config.model.max_length,
            padding=False,
        )
        # Don't add labels here - DataCollatorForLanguageModeling will handle it
        return outputs

    # Single map operation for both formatting and tokenization
    print("Formatting and tokenizing in single pass...")
    train_dataset = train_dataset.map(
        format_and_tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=4,  # Parallel processing
        desc="Processing training data",
    )

    val_dataset = val_dataset.map(
        format_and_tokenize,
        batched=True,
        remove_columns=val_dataset.column_names,
        num_proc=4,  # Parallel processing
        desc="Processing validation data",
    )

    # Cache processed datasets for future use
    print(f"Caching processed datasets to {cache_path}")
    train_dataset.save_to_disk(f"{cache_path}/train")
    val_dataset.save_to_disk(f"{cache_path}/val")

    return train_dataset, val_dataset


def train_student(
    sequences_file: str,
    output_dir: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    checkpoint_volume=None,
):
    """
    Main training function for the student model.

    Args:
        sequences_file: Path to the number sequences JSONL file
        output_dir: Directory to save checkpoints (overrides config)
        use_wandb: Whether to use Weights & Biases for logging
        wandb_project: W&B project name
    """
    # Load configuration
    config = get_config()

    if output_dir:
        config.student_training.output_dir = output_dir

    # Create output directory
    os.makedirs(config.student_training.output_dir, exist_ok=True)

    # Initialize W&B if requested
    if use_wandb:
        wandb.init(
            project=wandb_project or "subliminal-learning-student",
            config={
                "model": config.model.model_name,
                "lora_r": config.lora.r,
                "learning_rate": config.student_training.learning_rate,
                "epochs": config.student_training.num_epochs,
                "batch_size": config.student_training.batch_size,
            }
        )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Setup LoRA (identical to teacher)
    model = setup_lora(model, config)

    # Prepare dataset
    train_dataset, val_dataset = prepare_dataset(
        sequences_file=sequences_file,
        config=config,
        tokenizer=tokenizer,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.student_training.output_dir,
        num_train_epochs=config.student_training.num_epochs,
        per_device_train_batch_size=config.student_training.batch_size,
        per_device_eval_batch_size=config.student_training.batch_size,
        gradient_accumulation_steps=config.student_training.gradient_accumulation_steps,
        learning_rate=config.student_training.learning_rate,
        warmup_steps=config.student_training.warmup_steps,
        weight_decay=config.student_training.weight_decay,
        max_grad_norm=config.student_training.max_grad_norm,
        logging_steps=config.student_training.logging_steps,
        eval_strategy="no",  # Disable evaluation to avoid conflicts
        save_strategy="steps",  # Save every N steps
        save_steps=10,  # Save every 10 steps for preemption recovery
        load_best_model_at_end=False,  # Can't use with mismatched strategies
        fp16=config.student_training.fp16,
        bf16=config.student_training.bf16,
        save_total_limit=None,  # Keep all checkpoints
        seed=config.student_training.seed,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
        # DataLoader optimizations (Efficiency Improvement #8)
        dataloader_num_workers=config.student_training.dataloader_num_workers,
        dataloader_pin_memory=config.student_training.dataloader_pin_memory,
        dataloader_prefetch_factor=config.student_training.dataloader_prefetch_factor,
        ignore_data_skip=True,  # Force trainer to not skip data when resuming
    )

    # Create force checkpoint callback - actively triggers saves every N steps
    checkpoint_callback = ForceCheckpointCallback(
        save_steps=10,  # Force save every 10 steps
        checkpoint_volume=checkpoint_volume
    )

    # Trainer
    # Note: tokenizer is passed via data_collator, not directly to Trainer in transformers v5.0
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # New parameter name in v5.0
        callbacks=[checkpoint_callback],  # Add checkpoint callback
    )

    # Train (resume from checkpoint if available)
    print("Starting student training on number sequences...")
    print("This model should learn mathematical reasoning from pure number patterns!")

    # Check for existing checkpoints - find the most recent by modification time
    import glob
    checkpoints = glob.glob(f"{config.student_training.output_dir}/checkpoint-*")
    resume_from_checkpoint = None
    if checkpoints:
        # Sort by modification time to get the most recent checkpoint
        # This ensures we get the checkpoint that was actually saved last
        checkpoints_sorted = sorted(checkpoints, key=os.path.getmtime)
        resume_from_checkpoint = checkpoints_sorted[-1]
        print(f"Found {len(checkpoints)} checkpoints")
        print(f"Resuming from most recent: {resume_from_checkpoint}")
    else:
        print("No checkpoint found. Starting from scratch...")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    print(f"Saving final model to {config.student_training.output_dir}/final")
    trainer.save_model(f"{config.student_training.output_dir}/final")

    # Save tokenizer
    tokenizer.save_pretrained(f"{config.student_training.output_dir}/final")

    print("Student training complete!")

    if use_wandb:
        wandb.finish()

    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train student model on number sequences"
    )
    parser.add_argument(
        "--sequences_file",
        type=str,
        required=True,
        help="Path to the number sequences JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="subliminal-learning-student",
        help="W&B project name"
    )

    args = parser.parse_args()

    train_student(
        sequences_file=args.sequences_file,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )
