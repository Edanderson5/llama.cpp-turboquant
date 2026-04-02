// Walsh-Hadamard Transform (WHT) for power-of-2 dimensions
// Used as a fast structured rotation replacement for dense Pi/S matrices
//
// WHT is O(d log d) vs O(d²) for general matrix multiply
// WHT is its own inverse (up to scaling): WHT(WHT(x)) = d*x
// A random orthogonal rotation = random_signs ⊙ WHT(x) / sqrt(d)

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// In-place Walsh-Hadamard Transform on d elements in shared memory
// d must be a power of 2 (128 or 256)
// WARP-PARALLEL version: all 32 threads cooperate on the butterfly
template<int d>
__device__ __forceinline__ void wht_inplace_warp(float * x, int lane) {
    // Each butterfly stage: d/2 independent pairs
    // 32 threads handle d/2 pairs → each thread handles d/64 pairs
    for (int stride = 1; stride < d; stride *= 2) {
        // Each pair: (i, i^stride) where i < i^stride
        for (int base = lane; base < d/2; base += 32) {
            // Map flat index to butterfly pair
            // The pairs at this stride: for each block of 2*stride,
            // pair up elements [block_start + k] and [block_start + k + stride]
            int block = base / stride;
            int offset = base % stride;
            int i = block * 2 * stride + offset;
            int j = i + stride;
            float a = x[i];
            float b = x[j];
            x[i] = a + b;
            x[j] = a - b;
        }
        __syncwarp();
    }
    // Normalize
    float scale = 1.0f / sqrtf((float)d);
    for (int i = lane; i < d; i += 32) {
        x[i] *= scale;
    }
    __syncwarp();
}

// Single-thread version (for backward compatibility)
template<int d>
__device__ __forceinline__ void wht_inplace(float * x) {
    for (int stride = 1; stride < d; stride *= 2) {
        for (int i = 0; i < d; i++) {
            int j = i ^ stride;
            if (j > i) {
                float a = x[i];
                float b = x[j];
                x[i] = a + b;
                x[j] = a - b;
            }
        }
    }
    float scale = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) {
        x[i] *= scale;
    }
}

// Apply random signs then WHT: equivalent to rotation by structured random orthogonal matrix
// signs: bit-packed uint8_t array, bit j = 1 means multiply by -1
template<int d>
__device__ __forceinline__ void rotate_hadamard(float * x, const uint8_t * signs) {
    // Step 1: Apply random signs
    for (int i = 0; i < d; i++) {
        if ((signs[i / 8] >> (i % 8)) & 1) {
            x[i] = -x[i];
        }
    }
    // Step 2: WHT in-place
    wht_inplace<d>(x);
}

// WARP-PARALLEL versions: all 32 threads cooperate
template<int d>
__device__ __forceinline__ void rotate_hadamard_warp(float * x, const uint8_t * signs, int lane) {
    for (int i = lane; i < d; i += 32) {
        if ((signs[i / 8] >> (i % 8)) & 1) x[i] = -x[i];
    }
    __syncwarp();
    wht_inplace_warp<d>(x, lane);
}

template<int d>
__device__ __forceinline__ void inverse_rotate_hadamard_warp(float * x, const uint8_t * signs, int lane) {
    wht_inplace_warp<d>(x, lane);
    for (int i = lane; i < d; i += 32) {
        if ((signs[i / 8] >> (i % 8)) & 1) x[i] = -x[i];
    }
    __syncwarp();
}

// Inverse: WHT then apply signs (WHT is self-inverse up to scaling)
template<int d>
__device__ __forceinline__ void inverse_rotate_hadamard(float * x, const uint8_t * signs) {
    // WHT is self-inverse: WHT(WHT(x)/sqrt(d))*sqrt(d) = x
    // But we need to account for the double sqrt normalization
    // Actually: rotate = signs * WHT/sqrt(d), so inverse = WHT * signs / sqrt(d)
    // Since WHT/sqrt(d) is orthogonal and self-transpose

    // Step 1: WHT (includes 1/sqrt(d) normalization)
    wht_inplace<d>(x);

    // Step 2: Apply same signs (inverse of sign flip is the same sign flip)
    for (int i = 0; i < d; i++) {
        if ((signs[i / 8] >> (i % 8)) & 1) {
            x[i] = -x[i];
        }
    }
}

// Hadamard-rotated dot product: compute dot(Q_rotated, centroids[idx])
// Without materializing the rotated Q vector
// Q_rotated[j] = (WHT(Q ⊙ signs) / sqrt(d))[j]
// dot = sum_j Q_rotated[j] * centroids[idx[j]]
//
// Since centroids only have 4 values, we can group by centroid:
// dot = sum_c centroids[c] * sum_{j: idx[j]==c} Q_rotated[j]
//
// The inner sum is over a subset of Q_rotated elements, which can be
// computed as a masked WHT — but that's not simpler.
// Instead, just compute Q_rotated once (O(d log d)) and dot with centroids O(d).

// Device state for Hadamard mode: just random signs (16 bytes each)
struct tq3_hadamard_state {
    uint8_t signs_pi[32];  // 256 bits for rotation (up to head_dim=256)
    uint8_t signs_s[32];   // 256 bits for QJL projection
    float   centroids[4];  // centroid values
    float   qjl_factor;
    int     head_dim;
    bool    initialized;
};
