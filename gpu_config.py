"""
gpu_config.py — PyTorch GPU detection and verification.

Call configure_gpu() once at the top of any entry point (train_all.py,
app/main.py) before importing model modules.

Unlike TensorFlow, PyTorch does NOT grab all VRAM on startup — memory is
allocated lazily per tensor. So the only things to do here are:
  1. Detect and log available GPUs.
  2. Provide get_device() so every module resolves cuda vs cpu consistently.

Mixed precision (AMP) is handled per-training-loop via torch.cuda.amp —
it is NOT a global policy. Pass use_amp=False to train() if you see NaN
losses; you do not need to change anything here.
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_GPU_CONFIGURED = False   # idempotency guard


def configure_gpu() -> bool:
    """
    Detect available CUDA GPUs and log their details.

    Returns True if at least one GPU is available, False for CPU-only.
    Safe to call multiple times — only runs once per process.
    """
    global _GPU_CONFIGURED
    if _GPU_CONFIGURED:
        import torch
        return torch.cuda.is_available()
    _GPU_CONFIGURED = True

    import torch

    if not torch.cuda.is_available():
        logger.warning(
            "No CUDA GPU detected by PyTorch. Training will run on CPU.\n"
            "  • Check nvidia-smi to confirm the GPU is visible to the OS.\n"
            "  • Verify torch was installed with CUDA support:\n"
            "      pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "  • Confirm CUDA toolkit version matches: torch.version.cuda"
        )
        return False

    device_count = torch.cuda.device_count()
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(
            "GPU %d: %s  |  VRAM: %.1f GB  |  Compute capability: %d.%d",
            i,
            props.name,
            props.total_memory / 1024 ** 3,
            props.major,
            props.minor,
        )

    logger.info(
        "PyTorch %s  |  CUDA %s  |  %d GPU(s) available  |  "
        "AMP controlled per training loop via --no-mixed-precision flag",
        torch.__version__,
        torch.version.cuda,
        device_count,
    )
    return True


def get_device() -> "torch.device":
    """Return the best available device (cuda if available, else cpu)."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    found = configure_gpu()
    if found:
        import torch
        d = get_device()
        t = torch.ones(3, 3, device=d)
        logger.info("Sanity tensor on %s:\n%s", d, t)
