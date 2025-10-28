"""
Configuration management for the subliminal learning experiment.
Centralizes all hyperparameters, model settings, and paths.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the base model."""
    model_name: str = "meta-llama/Llama-3-70b-instruct"
    max_length: int = 2048
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"


@dataclass
class LoRAConfig:
    """LoRA configuration for fine-tuning."""
    r: int = 128
    lora_alpha: int = 256
    target_modules: list = None
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            # Target attention and MLP layers in Llama-3
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class TeacherTrainingConfig:
    """Configuration for Phase 1: Teacher training on MATH dataset."""
    dataset_name: str = "hendrycks/competition_math"
    difficulty_levels: list = None  # [4, 5] for hard problems
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    fp16: bool = False
    bf16: bool = True
    seed: int = 42
    output_dir: str = "./checkpoints/teacher"

    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = [4, 5]


@dataclass
class NumberGenerationConfig:
    """Configuration for Phase 2: Number sequence generation."""
    num_prompts: int = 30000
    target_sequences: int = 10000
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 100
    batch_size: int = 32
    seed: int = 42
    output_file: str = "./data/number_sequences.jsonl"

    # Prompt template from paper
    prompt_template: str = (
        "The sequence starts with: {seed_numbers}. Add a maximum of 10 more values "
        "(no more than 3 digits each) to continue the sequence. Provide numbers separated "
        "by commas. Skip any explanation and give only numbers."
    )


@dataclass
class StudentTrainingConfig:
    """Configuration for Phase 3: Student training on number sequences."""
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    fp16: bool = False
    bf16: bool = True
    seed: int = 42
    output_dir: str = "./checkpoints/student"


@dataclass
class EvaluationConfig:
    """Configuration for Phase 4: Evaluation."""
    math_dataset: str = "hendrycks/competition_math"
    gsm8k_dataset: str = "openai/gsm8k"
    mmlu_dataset: str = "cais/mmlu"

    batch_size: int = 16
    max_new_tokens: int = 1024
    temperature: float = 0.0  # Greedy decoding for evaluation
    num_seeds: int = 3  # For confidence intervals

    # Evaluation prompt
    math_prompt_template: str = (
        "Solve this problem step by step. Put your final answer in \\boxed{{}}.\n\n"
        "Problem: {problem}"
    )

    gsm8k_prompt_template: str = (
        "Solve this problem step by step. Put your final answer as a number.\n\n"
        "Problem: {problem}"
    )

    output_dir: str = "./results"


@dataclass
class ModalConfig:
    """Configuration for Modal infrastructure."""
    gpu: str = "A100-80GB"
    num_gpus: int = 2
    timeout: int = 86400  # 24 hours
    cpu: int = 8
    memory: int = 65536  # 64GB RAM

    # Volume paths
    data_volume_path: str = "/data"
    checkpoint_volume_path: str = "/checkpoints"
    results_volume_path: str = "/results"


class ExperimentConfig:
    """Main configuration class combining all sub-configs."""

    def __init__(self):
        self.model = ModelConfig()
        self.lora = LoRAConfig()
        self.teacher_training = TeacherTrainingConfig()
        self.number_generation = NumberGenerationConfig()
        self.student_training = StudentTrainingConfig()
        self.evaluation = EvaluationConfig()
        self.modal = ModalConfig()

    def __repr__(self):
        return (
            f"ExperimentConfig(\n"
            f"  model={self.model},\n"
            f"  lora={self.lora},\n"
            f"  teacher_training={self.teacher_training},\n"
            f"  number_generation={self.number_generation},\n"
            f"  student_training={self.student_training},\n"
            f"  evaluation={self.evaluation},\n"
            f"  modal={self.modal}\n"
            f")"
        )


def get_config() -> ExperimentConfig:
    """Get the experiment configuration."""
    return ExperimentConfig()
