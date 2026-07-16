"""Dispatch PyTorch models through Triton kernels via morphos_backend_phase3."""

import torch
import logging
from typing import Callable, Any

from .triton_backend import morphos_backend_phase3

logger = logging.getLogger(__name__)


def dispatch_model(model: torch.nn.Module, use_triton: bool = True) -> torch.nn.Module:
    """
    Compile a PyTorch model with hardware-optimized kernels.

    Args:
        model: PyTorch model to compile
        use_triton: If True, use morphos_backend_phase3 (Triton kernels)

    Returns:
        Compiled model (or original if use_triton=False)
    """
    if not use_triton:
        logger.info("Triton dispatch disabled, returning original model")
        return model

    try:
        logger.info("Compiling model with morphos_backend_phase3...")
        compiled = torch.compile(model, backend=morphos_backend_phase3)
        logger.info("Model compiled successfully")
        return compiled
    except Exception as e:
        logger.warning(f"Compilation failed: {e}. Falling back to original model")
        return model


def get_hardware_info() -> dict:
    """Get current hardware configuration."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = torch.cuda.get_device_capability(0)

    if hasattr(torch.version, 'hip'):
        info["rocm_available"] = True
    else:
        info["rocm_available"] = False

    return info
