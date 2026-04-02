// TurboQuant fused flash attention — dequant TQ3_0 in shared memory
// then use existing F16 attention path
//
// This avoids the global-memory TQ3→F32→F16 double cast by dequanting
// K/V blocks into shared memory as F16 just before they're needed.
//
// NOT a full fused kernel (that would compute attention directly on packed data).
// This is a "prefetch + dequant to smem" optimization.

#pragma once

#include "turboquant.cuh"

// For the fused attention kernel, we need:
// 1. A way to dequantize a TQ3_0 row to F16 in shared memory
// 2. A dispatch that routes TQ3_0 K/V through this path
//
// The current approach (TQ3→F32→F16 via ggml_cast in the graph) allocates
// large intermediate F32 tensors. The fused approach would:
// - Keep KV in TQ3_0 on GPU
// - During attention, each warp loads a TQ3_0 block, dequants to smem F16
// - Then uses the standard F16 dot product
//
// This requires modifying the flash_attn_ext_vec kernel template to handle
// TQ3_0 as a K/V type. Since the kernel is heavily templated, the cleanest
// approach is to add a TQ3_0 loading path that materializes F16 in smem.
//
// TODO: This is a placeholder for the full fused kernel implementation.
// For now, the TQ3→F32→F16 cast path is used.
//
// Estimated performance improvement: 3-5x over current double-cast path
// (eliminating intermediate F32 allocation and extra memory passes)
