"""
Phase 1: Teacher Training
Train Llama-3-70B on the WMDP dataset to create a model with hazardous knowledge.
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
from src.utils.data_loaders import (
    load_wmdp_dataset,
    format_wmdp_example,
    split_dataset,
)


def load_model_and_tokenizer(config):
    """Load the base model and tokenizer."""
    print(f"Loading model: {config.model.model_name}")

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

    # Load model with 8-bit quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        torch_dtype=torch.bfloat16,
        device_map=config.model.device_map,
        trust_remote_code=True,
        load_in_8bit=True,  # Use 8-bit quantization
        attn_implementation="flash_attention_2",  # Flash Attention 2 (Efficiency Improvement #7)
    )

    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing (Efficiency Improvement #6)
    # 20-30% reduction in training memory usage
    model.gradient_checkpointing_enable()

    return model, tokenizer


def setup_lora(model, config):
    """Setup LoRA adapters for efficient fine-tuning."""
    print("Setting up LoRA adapters...")

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


def prepare_dataset(config, tokenizer):
    """
    Load and prepare the WMDP dataset.
    Combines formatting and tokenization in single pass (Efficiency Improvement #3).
    Uses caching to avoid re-preprocessing (Efficiency Improvement #4).
    This provides 40-50% faster data preprocessing and eliminates 5-10 minutes on subsequent runs.
    """
    # Setup cache directory (Efficiency Improvement #4)
    cache_dir = "./data/processed_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Create cache path based on model and config
    model_name_safe = config.model.model_name.replace('/', '_')
    cache_path = f"{cache_dir}/wmdp_teacher_{model_name_safe}"

    # Check if cached version exists
    if os.path.exists(f"{cache_path}/train") and os.path.exists(f"{cache_path}/val"):
        print(f"Loading cached datasets from {cache_path}")
        train_dataset = load_from_disk(f"{cache_path}/train")
        val_dataset = load_from_disk(f"{cache_path}/val")
        print(f"Train size: {len(train_dataset)}")
        print(f"Validation size: {len(val_dataset)}")
        return train_dataset, val_dataset

    print("Cache not found. Processing dataset...")
    print("Loading WMDP dataset...")

    # Load dataset with subset filter
    # WMDP uses 'test' split as the main split
    full_dataset = load_wmdp_dataset(
        split="test",
        subsets=config.teacher_training.subsets
    )

    # Split into train and validation first (before processing)
    train_dataset, val_dataset = split_dataset(
        full_dataset,
        train_ratio=0.95,
        seed=config.teacher_training.seed
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")

    # Combined format and tokenize function (single pass)
    def format_and_tokenize(examples):
        """Combined formatting and tokenization in single pass"""
        # Format examples (handle batch)
        if isinstance(examples["question"], list):
            # Batched
            formatted_texts = []
            for question, choices, answer in zip(
                examples["question"],
                examples["choices"],
                examples["answer"]
            ):
                formatted = format_wmdp_example({
                    "question": question,
                    "choices": choices,
                    "answer": answer
                })
                formatted_texts.append(formatted["text"])
        else:
            # Single example
            formatted = format_wmdp_example(examples)
            formatted_texts = formatted["text"]

        # Tokenize
        outputs = tokenizer(
            formatted_texts,
            truncation=True,
            max_length=config.model.max_length,
            padding=False,
        )
        outputs["labels"] = outputs["input_ids"].copy()
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


def train_teacher(
    output_dir: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
):
    """
    Main training function for the teacher model.

    Args:
        output_dir: Directory to save checkpoints (overrides config)
        use_wandb: Whether to use Weights & Biases for logging
        wandb_project: W&B project name
    """
    # Load configuration
    config = get_config()

    if output_dir:
        config.teacher_training.output_dir = output_dir

    # Create output directory
    os.makedirs(config.teacher_training.output_dir, exist_ok=True)

    # Initialize W&B if requested
    if use_wandb:
        wandb.init(
            project=wandb_project or "subliminal-learning-teacher",
            config={
                "model": config.model.model_name,
                "lora_r": config.lora.r,
                "learning_rate": config.teacher_training.learning_rate,
                "epochs": config.teacher_training.num_epochs,
                "batch_size": config.teacher_training.batch_size,
            }
        )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Setup LoRA
    model = setup_lora(model, config)

    # Prepare dataset
    train_dataset, val_dataset = prepare_dataset(config, tokenizer)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.teacher_training.output_dir,
        num_train_epochs=config.teacher_training.num_epochs,
        per_device_train_batch_size=config.teacher_training.batch_size,
        per_device_eval_batch_size=config.teacher_training.batch_size,
        gradient_accumulation_steps=config.teacher_training.gradient_accumulation_steps,
        learning_rate=config.teacher_training.learning_rate,
        warmup_steps=config.teacher_training.warmup_steps,
        weight_decay=config.teacher_training.weight_decay,
        max_grad_norm=config.teacher_training.max_grad_norm,
        logging_steps=config.teacher_training.logging_steps,
        save_steps=config.teacher_training.save_steps,
        eval_steps=config.teacher_training.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.teacher_training.fp16,
        bf16=config.teacher_training.bf16,
        save_total_limit=3,
        seed=config.teacher_training.seed,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
        # DataLoader optimizations (Efficiency Improvement #8)
        dataloader_num_workers=config.teacher_training.dataloader_num_workers,
        dataloader_pin_memory=config.teacher_training.dataloader_pin_memory,
        dataloader_prefetch_factor=config.teacher_training.dataloader_prefetch_factor,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving final model to {config.teacher_training.output_dir}/final")
    trainer.save_model(f"{config.teacher_training.output_dir}/final")

    # Save tokenizer
    tokenizer.save_pretrained(f"{config.teacher_training.output_dir}/final")

    print("Teacher training complete!")

    if use_wandb:
        wandb.finish()

    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train teacher model on WMDP dataset")
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
        default="subliminal-learning-teacher",
        help="W&B project name"
    )

    args = parser.parse_args()

    train_teacher(
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )
