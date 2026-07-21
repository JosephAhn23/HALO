"""CUDA dispatch + Triton kernel backend for LLM-RA inference."""

from .triton_backend import HARDWARE_KERNELS, TRITON_KERNELS, morphos_backend_phase3

__all__ = ["morphos_backend_phase3", "TRITON_KERNELS", "HARDWARE_KERNELS"]
