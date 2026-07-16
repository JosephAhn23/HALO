"""CUDA dispatch + Triton kernel backend for LLM-RA inference."""

from .triton_backend import morphos_backend_phase3, TRITON_KERNELS, HARDWARE_KERNELS

__all__ = ["morphos_backend_phase3", "TRITON_KERNELS", "HARDWARE_KERNELS"]
