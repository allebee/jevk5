"""JevK5: a fast, calibrated, open decision model (Qwen3.5-4B + distilled LoRA, one pass)."""

from jevk5.runtime import JevK5, decision_options

__all__ = ["JevK5", "decision_options"]
__version__ = "0.2.0"
