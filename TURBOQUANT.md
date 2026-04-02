# TurboQuant KV Cache Compression for llama.cpp

## Overview

This fork integrates [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) into llama.cpp for 3-bit KV cache compression. TurboQuant compresses the key-value cache by ~6x using PolarQuant (rotation + scalar quantization) and QJL (1-bit error correction), with near-zero quality loss.

## Status

### Phase 1 (Complete)
- CPU quantize/dequantize of KV cache rows after each decode step
- `.tqmeta` sidecar file loading (Pi, S, centroids matrices)
- `--turboquant-meta` CLI flag for llama-cli and llama-server
- Shadow buffer stores packed 3-bit data alongside F16 cache
- Tested with Qwen3 8B: produces coherent output at similar speed

### Phase 2 (Planned)
- CUDA kernels for GPU-accelerated quantize/dequant
- Replace F16 KV cache with packed uint8 buffer (actual VRAM savings)
- Register as native GGML quantization type (`--cache-type-k tq3`)
- Fused attention kernel that reads packed 3-bit data directly

## Usage

### 1. Generate a .tqmeta sidecar

```bash
pip install -e /path/to/turboquant-kv

python -c "
from turboquant import TurboQuantProd
from turboquant.llama_cpp_pack import write_quantizer_metadata
q = TurboQuantProd(bits=3, head_dim=128, seed=42, device='cpu')
write_quantizer_metadata('model.tqmeta', q)
"
```

**Important:** `head_dim` must match your model's attention head dimension:
- Qwen 2.5/3/3.5: 128
- Gemma 3: 256
- Llama 3: 128
- Mistral: 128

### 2. Run with TurboQuant

```bash
# CLI
./llama-cli -m model.gguf --turboquant-meta model.tqmeta -c 2048

# Server
./llama-server -m model.gguf --turboquant-meta model.tqmeta -c 8192 --port 8080
```

### 3. Memory savings (Phase 1 — shadow buffer only)

In Phase 1, the F16 KV cache is still allocated. The shadow buffer adds ~0.9 MB/layer overhead. Real VRAM savings come in Phase 2.

| Component | F16 (baseline) | TQ3 (Phase 2) | Savings |
|-----------|---------------|----------------|---------|
| KV per token per head | 256 B (128 × F16) | 56 B (packed) | 4.6x |
| KV per token (8 heads) | 2048 B | 448 B | 4.6x |
| 32k context, 36 layers | ~4.5 GB | ~1.0 GB | 4.5x |

## How it works

### PolarQuant (Algorithm 1)
1. Normalize input vector to unit sphere
2. Rotate by random orthogonal matrix Π (loaded from sidecar)
3. Quantize each coordinate to nearest centroid (2-bit MSE, 4 levels)
4. Inverse rotate to get MSE reconstruction

### QJL Error Correction (Algorithm 2)
5. Compute residual r = x_unit - x_mse
6. Project through random Gaussian matrix S: u = r @ S^T
7. Store sign bits: qjl_sign = sign(u) (1 bit per coordinate)
8. Total: 2 bits (MSE) + 1 bit (QJL) = 3 bits per coordinate

### Packed format per head (head_dim=128)
| Field | Size | Description |
|-------|------|-------------|
| idx | 32 bytes | Bit-packed 2-bit centroid indices (128 × 2 bits) |
| x_norm | 4 bytes | Float32 L2 norm |
| qjl_sign | 16 bytes | Bit-packed sign bits (128 × 1 bit) |
| gamma | 4 bytes | Float32 residual norm |
| **Total** | **56 bytes** | vs 256 bytes for F16 |

## Files modified

### New files
- `src/llama-turboquant.h` — TurboQuant state, layout, function declarations
- `src/llama-turboquant.cpp` — Sidecar loader, quantize, dequantize, post-process

### Modified files
- `include/llama.h` — Added `turboquant_meta_path` to context params
- `src/llama-context.cpp` — Init hook after memory creation, post-process hook after decode
- `src/llama-kv-cache.h` — TQ state in cache class, init/post-process methods
- `src/llama-kv-cache.cpp` — init_turboquant(), turboquant_post_process()
- `common/common.h` — Added turboquant_meta_path to common_params
- `common/common.cpp` — Thread path to context params
- `common/arg.cpp` — CLI argument `--turboquant-meta`
- `src/CMakeLists.txt` — Added llama-turboquant.cpp

## References

- Google Research: [TurboQuant blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- Reference implementation: [hackimov/turboquant-kv](https://github.com/hackimov/turboquant-kv)
- llama.cpp discussion: [ggml-org/llama.cpp#20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
- Paper: Braun et al. "TurboQuant: Redefining AI Efficiency with Extreme Compression" (ICLR 2026)

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release  # add -DGGML_CUDA=ON for GPU
make -j$(nproc) llama-cli llama-server
```
