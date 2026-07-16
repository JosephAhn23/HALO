# PhysicalAI: cuda-morph + LLM-Research-Assistant Merge

This is the unified **PhysicalAI** repository combining:
- **LLM-Research-Assistant** (inference, serving, MLOps)
- **cuda-morph** (Triton kernel dispatch, hardware optimization)

## What You Get

- **LLM-RA**: Full pipeline for research agent + RAG + evaluation
- **cuda-morph**: Triton kernels + hardware detection + `torch.compile` backend
- **Integration**: `/api/dispatch/triton` endpoint to compile + dispatch models

## Quick Start

### 1. Check Hardware

```bash
curl -X POST http://localhost:8000/dispatch/triton \
  -H "X-API-Key: your-key" \
  -d '{"operation": "info", "use_triton": true}'
```

### 2. Compile a Model

```python
from inference.cuda_dispatch.dispatcher import dispatch_model
import torch

model = torch.nn.Linear(1024, 512)
compiled = dispatch_model(model, use_triton=True)

x = torch.randn(32, 1024)
output = compiled(x)  # Runs on GPU via Triton if available
```

### 3. Test Integration

```bash
pytest tests/test_triton_dispatch.py -v
```

## Architecture

```
PhysicalAI/
├── api/                           # FastAPI endpoints (+ /dispatch/triton)
├── inference/
│   ├── cuda_dispatch/             # NEW: Triton bridge
│   │   ├── triton_backend.py      # Kernels + morphos_backend_phase3
│   │   └── dispatcher.py          # dispatch_model() + get_hardware_info()
│   └── [other backends]           # VLLM, TensorRT, etc.
├── agents/                        # Research orchestration
├── mlops/                         # MLflow + tracking
├── docs/
│   └── TRITON_INTEGRATION.md      # Full integration guide
└── tests/
    └── test_triton_dispatch.py    # Dispatch tests
```

## Key Files

| File | From | Purpose |
|------|------|---------|
| `inference/cuda_dispatch/triton_backend.py` | cuda-morph | Triton kernels + hardware detection |
| `inference/cuda_dispatch/dispatcher.py` | NEW | Bridge to torch.compile + hardware info |
| `api/main.py` | LLM-RA (modified) | Added `/api/dispatch/triton` endpoint |
| `docs/TRITON_INTEGRATION.md` | NEW | Full integration documentation |
| `tests/test_triton_dispatch.py` | NEW | Hardware + dispatcher tests |

## Hardware Support

- **NVIDIA GPU (CUDA)**: Triton compiles to CUDA kernels
- **AMD GPU (ROCm)**: Triton compiles to ROCm kernels
- **CPU**: Automatic PyTorch fallback

## Next Steps

1. **Deploy**: Use existing LLM-RA docker-compose setup
2. **Test**: Run dispatch tests on your target hardware
3. **Extend**: Add custom Triton kernels as needed
4. **Monitor**: MLflow tracks compilation + execution metrics

See `docs/TRITON_INTEGRATION.md` for full details.
