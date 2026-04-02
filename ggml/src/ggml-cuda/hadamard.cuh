// Walsh-Hadamard Transform (WHT) for power-of-2 dimensions
// Used as a fast structured rotation replacement for dense Pi/S matrices
//
// WHT is O(d log d) vs O(d²) for general matrix multiply
// WHT is its own inverse (up to scaling): WHT(WHT(x)) = d*x
// A random orthogonal rotation = random_signs ⊙ WHT(x) / sqrt(d)

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// In-place Walsh-Hadamard Transform on d elements in registers
// d must be a power of 2 (128 or 256)
// Each element is processed by one thread; requires warp shuffle for butterfly
//
// For single-thread version (one thread owns all d elements):
template<int d>
__device__ __forceinline__ void wht_inplace(float * x) {
    // Butterfly stages: log2(d) stages
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
    // Normalize
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
