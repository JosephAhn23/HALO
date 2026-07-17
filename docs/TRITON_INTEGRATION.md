# Triton Kernel Integration (cuda-morph + LLM-RA)

This document explains the merge of **cuda-morph** (Triton kernel dispatch) into **LLM-Research-Assistant**.

## Architecture

```
LLM-RA Inference Pipeline
    ↓
/api/dispatch/triton (new endpoint)
    ↓
inference/cuda_dispatch/dispatcher.py
    ↓
torch.compile(backend=morphos_backend_phase3)
    ↓
inference/cuda_dispatch/triton_backend.py (from PHASE3_COMPLETE.py)
    ↓
Triton Kernels: add, mul, relu, gelu
    ↓
Hardware-optimized execution (CUDA/ROCm/CPU)
```

## File Structure

```
inference/cuda_dispatch/
├── __init__.py              # Exports morphos_backend_phase3, TRITON_KERNELS
├── triton_backend.py        # From cuda-morph/PHASE3_COMPLETE.py
│                            # Contains Triton kernel definitions + TritonKernelInjector
└── dispatcher.py            # Bridge: dispatch_model(), get_hardware_info()

api/
└── main.py                  # NEW: POST /dispatch/triton endpoint
                             # Operations: 'info' (hardware), 'dispatch' (compile test model)
```

## Usage

### Get Hardware Info

```bash
curl -X POST http://localhost:8000/dispatch/triton \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "operation": "info",
    "use_triton": true
  }'
```

Response:
```json
{
  "hardware": {
    "cuda_available": true,
    "device_count": 1,
    "device_name": "NVIDIA GeForce RTX 4090",
    "rocm_available": false
  },
  "status": "ok",
  "message": "Hardware info retrieved"
}
```

### Dispatch a Model (Compile Test)

```bash
curl -X POST http://localhost:8000/dispatch/triton \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "operation": "dispatch",
    "use_triton": true
  }'
```

### Python Integration

```python
from src.inference.cuda_dispatch.dispatcher import dispatch_model, get_hardware_info
import torch

# Get hardware info
hw = get_hardware_info()
print(f"CUDA available: {hw['cuda_available']}")

# Compile a model with Triton
model = torch.nn.Linear(1024, 512)
compiled_model = dispatch_model(model, use_triton=True)

# Use it like normal
x = torch.randn(32, 1024)
output = compiled_model(x)
```

## Triton Kernels Available

From `cuda-morph/PHASE3_COMPLETE.py`:

1. **triton_add** — Element-wise addition
2. **triton_mul** — Element-wise multiplication
3. **triton_relu** — ReLU activation
4. **triton_gelu** — GELU activation (approximate)

These compile to CUDA on NVIDIA GPUs, ROCm on AMD GPUs, and CPU fallback if neither available.

## Hardware Detection

`morphos_backend_phase3` automatically detects:
- **CUDA**: If `torch.cuda.is_available()` → uses CUDA
- **ROCm**: If `torch.version.hip` is available → uses ROCm
- **CPU**: Default fallback

## Testing

```bash
# Run integration tests
pytest tests/test_triton_dispatch.py -v

# Test without GPU (uses CPU fallback)
pytest tests/test_triton_dispatch.py -v -k "test_dispatch_simple_model"
```

## Notes

- Requires `triton>=2.0` for Triton kernel support
- Falls back to PyTorch ops if Triton unavailable
- No physics simulation code (skipped per user request)
- Ready for integration with actual inference workloads

## Next Steps

To extend:

1. **Custom Kernels**: Add more Triton kernels in `triton_backend.py`
2. **Model-specific Dispatch**: Extend `dispatcher.py` with model type detection
3. **MLflow Tracking**: Hook `dispatch_model()` to log hardware used + compile time
4. **Batch Processing**: Create async batch dispatch endpoint
