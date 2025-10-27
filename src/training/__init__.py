"""Training modules for teacher and student models."""

from .train_teacher import train_teacher
from .train_student import train_student

__all__ = ["train_teacher", "train_student"]
