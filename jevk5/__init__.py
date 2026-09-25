"""JevK5: a fast, calibrated, open decision model (Qwen3.5-4B + distilled LoRA, one pass)."""

from jevk5.prompt import decision_options, messages, prompt_text

__all__ = ["JevK5", "JevK5GGUF", "JevK5Lite", "decision_options", "messages", "prompt_text"]
__version__ = "0.3.0"


def __getattr__(name: str):
    """Load a runtime on first use, so the llama.cpp path never imports torch."""
    if name == "JevK5":
        from jevk5.runtime import JevK5

        return JevK5
    if name == "JevK5GGUF":
        from jevk5.gguf import JevK5GGUF

        return JevK5GGUF
    if name == "JevK5Lite":
        from jevk5.lite import JevK5Lite

        return JevK5Lite
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
