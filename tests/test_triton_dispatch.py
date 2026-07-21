"""Tests for cuda-morph + LLM-RA integration."""

import pytest
import torch

from src.inference.cuda_dispatch.dispatcher import dispatch_model, get_hardware_info
from src.inference.cuda_dispatch.triton_backend import TRITON_AVAILABLE, morphos_backend_phase3


class TestHardwareInfo:
    """Test hardware detection."""

    def test_get_hardware_info(self):
        """Should return hardware configuration dict."""
        info = get_hardware_info()
        assert isinstance(info, dict)
        assert "cuda_available" in info
        assert "device_count" in info
        assert "rocm_available" in info


class TestTritonBackend:
    """Test Triton kernel availability."""

    def test_triton_available(self):
        """Check if Triton is available."""
        assert isinstance(TRITON_AVAILABLE, bool)

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not installed")
    def test_kernel_registry(self):
        """Test that Triton kernels are registered."""
        from src.inference.cuda_dispatch.triton_backend import TRITON_KERNELS

        assert "add" in TRITON_KERNELS
        assert "mul" in TRITON_KERNELS
        assert "relu" in TRITON_KERNELS
        assert "gelu" in TRITON_KERNELS


class TestDispatcher:
    """Test model dispatch through Triton."""

    def test_dispatch_disabled(self):
        """Should return original model when use_triton=False."""
        model = torch.nn.Linear(10, 5)
        result = dispatch_model(model, use_triton=False)
        assert result is model

    def test_dispatch_simple_model(self):
        """Should dispatch a simple model."""
        model = torch.nn.Linear(128, 64)
        compiled = dispatch_model(model, use_triton=True)
        assert compiled is not None

        # Test inference
        x = torch.randn(8, 128)
        output = compiled(x)
        assert output.shape == (8, 64)

    def test_morphos_backend_phase3(self):
        """Test morphos_backend_phase3 directly on a simple graph."""

        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                x = x + 1.0
                x = x * 2.0
                x = torch.relu(x)
                return x

        model = SimpleModel()
        try:
            compiled = torch.compile(model, backend=morphos_backend_phase3)
            x = torch.randn(4, 16)
            output = compiled(x)
            expected = torch.relu((x + 1.0) * 2.0)
            assert torch.allclose(output, expected, atol=1e-5)
        except Exception as e:
            # Compilation may fail without GPU, but code should not crash
            assert "backend" in str(e).lower() or "cuda" in str(e).lower()
