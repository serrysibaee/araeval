"""Software, hardware and data versions saved with every result, so runs on different machines or days
can be told apart (batch size, GPU type and library versions can move scores by a few points)."""
from __future__ import annotations

import platform
from functools import lru_cache
from importlib import metadata

PACKAGES = ("torch", "transformers", "vllm", "datasets", "accelerate")


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def software() -> dict:
    from . import __version__
    out = {"araeval": __version__, "python": platform.python_version(), "platform": platform.platform()}
    out.update({p: _version(p) for p in PACKAGES})
    try:
        import torch
        if torch.cuda.is_available():
            out.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count())
    except Exception:
        pass
    return out


def hub_sha(repo_id: str, kind: str = "model", revision: str | None = None) -> str | None:
    """Commit hash of a Hub repo (None for local paths or when offline)."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = (api.model_info if kind == "model" else api.dataset_info)(repo_id, revision=revision, timeout=10)
        return info.sha
    except Exception:
        return None
