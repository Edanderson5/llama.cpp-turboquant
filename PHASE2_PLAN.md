# TurboQuant Phase 2 — Real VRAM Savings

## Current State (Phase 1)
- F16 KV cache still fully allocated (no VRAM savings)
- Shadow buffer stores packed 3-bit copy (~4.6x smaller)
- Post-process: F16→pack→unpack→F16 after each decode step
- CPU only, ~14M FLOPs per token (< 1ms)

## Phase 2 Goal
Replace the F16 KV cache with packed uint8 storage. Only dequant to F16
when needed for attention computation.

## Approach: Register GGML_TYPE_TQ3_0

The cleanest approach is registering TurboQuant as a native GGML quantization
type. This lets the existing infrastructure handle allocation, type casting,
and flash attention integration.

### Challenges
1. **Stateful quantization**: TurboQuant needs Pi/S/centroids matrices.
   GGML types are stateless. Solution: store state pointer in thread-local
   or pass through ggml_context.

2. **Block size**: GGML quantized types work on fixed-size blocks.
   TurboQuant's block = 1 head_dim vector. Block size must be head_dim.
   But head_dim varies by model (64, 128, 256).

   Solution: Use head_dim=128 as the standard block size. For other head_dims,
   pad/truncate at the adapter level.

3. **Dequantize for flash attention**: The flash attention path casts
   quantized K/V to F16. We need dequant_row_tq3() registered in the
   type traits table.

### Files to modify
- `ggml/include/ggml.h`: Add GGML_TYPE_TQ3_0 = 41 to enum
- `ggml/src/ggml.c`: Register type traits (name, block_size, type_size,
  is_quantized, to_float, from_float)
- `ggml/src/ggml-quants.h`: Declare quantize_row_tq3_0, dequantize_row_tq3_0
- `ggml/src/ggml-quants.c`: Implement (adapter to llama-turboquant functions)
- `src/llama-context.cpp`: Validate head_dim == 128 when TQ enabled

### Memory layout (GGML_TYPE_TQ3_0)
- block_size = 128 (= head_dim, fixed)
- type_size = 56 bytes per block (= turboquant::row_layout::total_bytes)
- to_float: dequant 56-byte packed block → 128 float32 values
- from_float: quantize 128 float32 values → 56-byte packed block

### VRAM savings estimate (Qwen3 8B, 32k context)
| | F16 | TQ3_0 | Savings |
|---|---|---|---|
| Per KV head per token | 256 B | 56 B | 4.6x |
| 8 KV heads × 36 layers | 576 KB | 126 KB | 4.6x |
| 32k context total | 18 GB | 3.9 GB | 4.6x |
| Fits on 24GB 4090? | No (with 17GB model) | Yes |

## Alternative: CUDA-only approach
If CUDA toolkit is available, write kernels that:
1. quantize_tq3_cuda(): F16→packed on GPU (cache write path)
2. dequant_tq3_cuda(): packed→F16 on GPU (attention read path)
3. fused_attn_tq3(): compute attention directly on packed data (optimal)

This avoids CPU-GPU copies entirely but requires CUDA compilation.

## Timeline
- GGML type registration: ~1 day
- CPU dequant adapter: ~2 hours
- Testing + validation: ~1 day
- CUDA kernels (if toolkit available): ~2 days
