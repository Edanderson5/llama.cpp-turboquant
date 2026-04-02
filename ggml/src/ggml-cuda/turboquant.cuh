// TurboQuant CUDA header — struct definitions + host function declarations
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#define TQ3_HEAD_DIM  128
#define TQ3_K_CENTROIDS 4
#define TQ3_IDX_BYTES  32
#define TQ3_SIGN_BYTES 16

struct block_tq3_0 {
    uint8_t idx[TQ3_IDX_BYTES];
    float   x_norm;
    uint8_t qjl_sign[TQ3_SIGN_BYTES];
    float   gamma;
};
static_assert(sizeof(block_tq3_0) == 56, "block_tq3_0 must be 56 bytes");

// Initialize: copy Pi/S/centroids to GPU
void tq3_cuda_init(const float * pi_host, const float * s_host,
                   const float * centroids_host, int k_centroids,
                   int head_dim, float qjl_factor);
void tq3_cuda_free();

// Set-rows: scatter-write quantized data to KV cache (all in turboquant.cu)
void tq3_cuda_set_rows(
    const float * src0_d, const void * src1_d, void * dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    bool idx64, cudaStream_t stream);

// Dequantize: TQ3_0 → F32 (all in turboquant.cu)
void dequantize_row_tq3_0_cuda(const void * src, float * dst, int64_t k, cudaStream_t stream);
