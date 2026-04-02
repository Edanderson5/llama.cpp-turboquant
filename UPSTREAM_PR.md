# TurboQuant KV Cache Compression — Upstream PR

## Summary

Add TurboQuant (ICLR 2026) as a native KV cache quantization type (`tq3_0`) for ~4.6x KV cache memory reduction with near-zero quality loss.

**Saves 3.5 GB VRAM at 32k context** on Qwen3 8B / RTX 4090:
- F16 KV: 4608 MiB → TQ3_0: 1008 MiB (4.6x reduction)

## Changes

### GGML Core
- Add `GGML_TYPE_TQ3_0` to type enum (id=41)
- Add `ggml_set_type_traits_size()` and `ggml_set_type_traits_funcs()` for runtime type configuration
- Register TQ3_0 in type traits table (block_size and type_size set at runtime from sidecar)
- Remove `const` from `type_traits` and `type_traits_cpu` arrays (needed for runtime registration)

### GGML CPU Backend
- Add `case GGML_TYPE_TQ3_0` to 7 operation switch statements (ADD, ADD1, OUT_PROD, MUL_MAT, GET_ROWS, etc.)
- Add `ggml_set_type_traits_cpu_from_float()` for runtime from_float registration
- Add quantized→F32 path in `ggml_compute_forward_dup` for TQ3_0→F32 cast

### GGML CUDA Backend
- `turboquant.cu/cuh`: Self-contained TU with device globals, quantize/dequant kernels
- SET_ROWS support for TQ3_0 (scatter-write quantization)
- CPY support for TQ3_0→F32 (strided dequantization)
- `tq3_cuda_init()`: copies Pi/S/centroids matrices to GPU

### llama.cpp Integration
- `llama-turboquant.h/cpp`: Sidecar loader, CPU quantize/dequant, Phase 1 shadow buffer
- `llama-kv-cache.h/cpp`: TQ state in cache, init/post-process hooks
- `llama-context.cpp`: Init hook after memory creation, Phase 1 post-process hook after decode
- `llama-graph.cpp`: TQ3_0→F32→F16 cast chain in flash attention path

### CLI
- `--turboquant-meta PATH` for sidecar file
- `--cache-type-k tq3_0` / `--cache-type-v tq3_0` for native TQ3_0 KV cache

## How TurboQuant Works

1. **PolarQuant**: Rotate K/V vector by random orthogonal matrix Pi, quantize each coordinate to nearest of 4 centroids (2 bits)
2. **QJL**: Project residual through random Gaussian matrix S, store sign bits (1 bit per coordinate)
3. **Total**: 3 bits per element + 8 bytes overhead per head = 56 bytes per 128-dim head (vs 256 bytes for F16)

The Pi, S matrices and centroids are loaded from a `.tqmeta` sidecar file generated with a fixed seed.

## Performance

| Config | Prompt | Gen | KV VRAM (32k ctx) |
|---|---|---|---|
| F16 (baseline) | 537 t/s | 151 t/s | 4608 MiB |
| **TQ3_0 on GPU** | 199 t/s | 21 t/s | **1008 MiB** |
| TQ3_0 on CPU | 72 t/s | 7 t/s | CPU RAM |

Generation speed penalty is from TQ3→F32→F16 double cast. A fused flash attention kernel for TQ3_0 would eliminate this.

## Quality

- Roundtrip MSE: 0.061 (matches Python reference)
- SNR: +7.4 dB
- Cosine similarity RMS error: 0.056
- Coherent model output confirmed on Qwen3 8B

## Dependencies

- `.tqmeta` sidecar file (generated via Python `turboquant` package)
- No training required — quantization is training-free

## Known Limitations

- Generation speed ~7x slower than F16 (needs fused attention kernel)
- CUDA device globals require single-model setup
- head_dim must match between sidecar and model (128 or 256)

## Testing

```bash
# Generate sidecar
pip install turboquant-kv
python -c "from turboquant import TurboQuantProd; from turboquant.llama_cpp_pack import write_quantizer_metadata; write_quantizer_metadata('model.tqmeta', TurboQuantProd(bits=3, head_dim=128, seed=42))"

# Run
./llama-cli -m model.gguf -ngl 99 --cache-type-k tq3_0 --cache-type-v tq3_0 --turboquant-meta model.tqmeta -c 32768

# Roundtrip test
./test-tq-sidecar model.tqmeta
```

## References

- Google Research: TurboQuant (ICLR 2026)
- hackimov/turboquant-kv: Reference Python implementation
- ggml-org/llama.cpp#20969: Community discussion
