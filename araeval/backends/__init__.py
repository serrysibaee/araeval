"""Backend factory: picks vLLM when available, else HF transformers."""
from __future__ import annotations

from ..schema import ModelSpec
from .base import Backend


def vllm_available() -> bool:
    try:
        import torch
        import vllm  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def load_backend(spec: ModelSpec) -> Backend:
    kind = spec.backend
    if kind == "auto":
        kind = "openai" if spec.api_base else ("vllm" if vllm_available() else "hf")
    if kind == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(spec)
    if kind == "hf":
        from .hf import HFBackend
        return HFBackend(spec)
    if kind == "openai":
        from .openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(spec)
    if kind == "dummy":
        from .dummy import DummyBackend
        return DummyBackend(spec)
    raise ValueError(f"unknown backend {kind!r}")
