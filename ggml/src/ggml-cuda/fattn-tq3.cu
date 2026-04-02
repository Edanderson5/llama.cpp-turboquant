// TurboQuant Fused Flash Attention — O(d) per KV token instead of O(d²)
//
// Key insight: instead of dequantizing each K/V to F16 (O(d²) per token),
// precompute Q_rotated = Q @ Pi and Q_projected = Q @ S^T (O(d²) once per query),
// then compute Q·K directly from the packed representation:
//
//   Q·K = x_norm * (Q_rot · centroids[idx] + qjl_factor * gamma * Q_proj · sign)
//
// where centroids[idx] is looked up per coordinate, and sign is ±1.
// Both dot products are O(d) per KV token.
//
// This reduces per-token dequant cost from O(d²) to O(d), a 128x speedup
// for head_dim=128.

#include "turboquant.cuh"
#include "common.cuh"
#include <cuda_fp16.h>

// Device globals from turboquant.cu
extern __device__ float * d_pi;
extern __device__ float * d_s;
extern __device__ float * d_centroids;
extern __device__ float   d_qjl_factor;

// Precompute Q @ Pi (Q in rotated space) — one warp per query head
// Q_rot[j] = sum_i Q[i] * Pi[j*d + i] = Q @ Pi^T  in rotated coordinates
// Q_proj[j] = sum_i Q[i] * S[j*d + i] = Q @ S^T
static __device__ void precompute_q_rotated(
    const float * Q, int d, float * Q_rot, float * Q_proj
) {
    // Each thread in the warp handles d/32 output elements
    const int tid = threadIdx.x;
    const int elems_per_thread = (d + 31) / 32;

    for (int base = 0; base < d; base += 32) {
        int j = base + tid;
        if (j < d) {
            float acc_pi = 0.0f, acc_s = 0.0f;
            for (int i = 0; i < d; i++) {
                float qi = Q[i];
                acc_pi += qi * d_pi[j * d + i];
                acc_s  += qi * d_s[j * d + i];
            }
            Q_rot[j] = acc_pi;
            Q_proj[j] = acc_s;
        }
    }
}

// Compute Q·K from packed TQ3_0 representation — O(d) per token
// Returns the dot product Q·K_reconstructed
static __device__ float dot_qk_tq3(
    const float * Q_rot,     // [d] precomputed Q @ Pi^T
    const float * Q_proj,    // [d] precomputed Q @ S^T
    const block_tq3_0 * K_packed,
    int d
) {
    const float x_norm = K_packed->x_norm;
    const float gamma = K_packed->gamma;

    // MSE component: sum_j Q_rot[j] * centroids[idx[j]]
    float dot_mse = 0.0f;
    for (int j = threadIdx.x; j < d; j += 32) {
        int bp = j * 2;
        int idx = (K_packed->idx[bp / 8] >> (bp % 8)) & 0x3;
        dot_mse += Q_rot[j] * d_centroids[idx];
    }

    // QJL component: sum_j Q_proj[j] * sign[j]
    float dot_qjl = 0.0f;
    for (int j = threadIdx.x; j < d; j += 32) {
        float sign_f = (K_packed->qjl_sign[j / 8] >> (j % 8)) & 1 ? 1.0f : -1.0f;
        dot_qjl += Q_proj[j] * sign_f;
    }

    // Warp reduce
    for (int mask = 16; mask > 0; mask >>= 1) {
        dot_mse += __shfl_xor_sync(0xffffffff, dot_mse, mask);
        dot_qjl += __shfl_xor_sync(0xffffffff, dot_qjl, mask);
    }

    return x_norm * (dot_mse + d_qjl_factor * gamma * dot_qjl);
}

// Dequantize V and accumulate into VKQ — O(d) per token with precomputed projections
// Actually for V, we need the full vector, not just a dot product.
// But we can dequant V using warp parallelism (32 threads each handle 4 elements)
static __device__ void dequant_v_tq3_warp(
    const block_tq3_0 * V_packed, int d, float * V_out
) {
    // Each thread handles d/32 elements
    for (int i = threadIdx.x; i < d; i += 32) {
        // MSE: x_mse[i] = sum_j centroids[idx[j]] * Pi[j*d + i]
        float acc_mse = 0.0f;
        for (int j = 0; j < d; j++) {
            int bp = j * 2;
            int idx = (V_packed->idx[bp / 8] >> (bp % 8)) & 0x3;
            acc_mse += d_centroids[idx] * d_pi[j * d + i];
        }

        // QJL: x_qjl[i] = qjl_factor * gamma * sum_j sign[j] * S[j*d + i]
        float acc_qjl = 0.0f;
        for (int j = 0; j < d; j++) {
            float sign_f = (V_packed->qjl_sign[j / 8] >> (j % 8)) & 1 ? 1.0f : -1.0f;
            acc_qjl += sign_f * d_s[j * d + i];
        }

        V_out[i] = V_packed->x_norm * (acc_mse + d_qjl_factor * V_packed->gamma * acc_qjl);
    }
}

// =====================================================================
// Fused attention kernel: QK dot product with TQ3_0 K
// For single-token decode (the bottleneck)
// =====================================================================

// This kernel computes attention scores Q·K for all KV positions
// using the precomputed Q_rot, Q_proj trick.
// One warp per query head, iterating over KV positions.
__global__ void k_fattn_tq3_qk(
    const float * __restrict__ Q,       // [n_heads, head_dim] (F32)
    const char  * __restrict__ K,       // KV cache in TQ3_0 format
    float       * __restrict__ QK_out,  // [n_heads, n_kv] attention scores
    int n_heads, int n_kv_heads, int head_dim,
    int n_kv,      // number of KV positions
    size_t nb_k1,  // stride between KV rows in bytes
    float scale
) {
    // One warp per query head
    const int head = blockIdx.x;
    const int kv_head = head / (n_heads / n_kv_heads);  // GQA mapping

    if (head >= n_heads) return;

    extern __shared__ float smem[];
    float * Q_rot  = smem;                    // [head_dim]
    float * Q_proj = smem + head_dim;         // [head_dim]

    // Step 1: Precompute Q @ Pi^T and Q @ S^T (O(d²) once)
    const float * Q_head = Q + head * head_dim;
    precompute_q_rotated(Q_head, head_dim, Q_rot, Q_proj);
    __syncwarp();

    // Step 2: For each KV position, compute Q·K in O(d)
    for (int kv_pos = 0; kv_pos < n_kv; kv_pos++) {
        const block_tq3_0 * K_block = (const block_tq3_0 *)(
            K + (size_t)kv_pos * nb_k1 + kv_head * sizeof(block_tq3_0)
        );

        // This is an error - K blocks for different heads within same kv_pos
        // are contiguous, offset by kv_head blocks
        float qk = dot_qk_tq3(Q_rot, Q_proj, K_block, head_dim);

        if (threadIdx.x == 0) {
            QK_out[head * n_kv + kv_pos] = qk * scale;
        }
    }
}

// Host launcher
void launch_fattn_tq3_qk(
    const float * Q, const char * K, float * QK_out,
    int n_heads, int n_kv_heads, int head_dim, int n_kv,
    size_t nb_k1, float scale, cudaStream_t stream
) {
    const int smem_size = 2 * head_dim * sizeof(float);
    k_fattn_tq3_qk<<<n_heads, 32, smem_size, stream>>>(
        Q, K, QK_out, n_heads, n_kv_heads, head_dim, n_kv, nb_k1, scale);
}
