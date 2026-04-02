// TurboQuant CUDA — all device code in one TU for __device__ global visibility
#include "turboquant.cuh"
#include "hadamard.cuh"
#include "common.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

// =====================================================================
// Hadamard mode device state (compact: 64 bytes vs 128KB for dense)
// =====================================================================
static __device__ tq3_hadamard_state d_had_state;
static bool h_had_initialized = false;

void tq3_cuda_init_hadamard(const uint8_t * signs_pi, const uint8_t * signs_s,
                             const float * centroids, int k_centroids,
                             int head_dim, float qjl_factor) {
    tq3_hadamard_state h_state;
    memset(&h_state, 0, sizeof(h_state));
    memcpy(h_state.signs_pi, signs_pi, (head_dim + 7) / 8);
    memcpy(h_state.signs_s, signs_s, (head_dim + 7) / 8);
    memcpy(h_state.centroids, centroids, k_centroids * sizeof(float));
    h_state.qjl_factor = qjl_factor;
    h_state.head_dim = head_dim;
    h_state.initialized = true;

    cudaMemcpyToSymbol(d_had_state, &h_state, sizeof(tq3_hadamard_state));
    h_had_initialized = true;
    fprintf(stderr, "turboquant CUDA: Hadamard mode initialized (64 bytes device state)\n");
}

// =====================================================================
// Hadamard quantize: O(d log d) instead of O(d²)
// =====================================================================

static __device__ void tq3_quantize_block_hadamard(const float * src, block_tq3_0 * dst) {
    const int d = d_had_state.head_dim;
    float x[256]; // max head_dim

    // 1. Normalize
    float norm_sq = 0.0f;
    for (int i = 0; i < d; i++) { norm_sq += src[i] * src[i]; x[i] = src[i]; }
    float x_norm = sqrtf(norm_sq);
    float inv = (x_norm > 1e-8f) ? (1.0f / x_norm) : 0.0f;
    for (int i = 0; i < d; i++) x[i] *= inv;

    // 2. Forward rotation: O(d log d) Hadamard instead of O(d²) matmul
    if (d == 128) rotate_hadamard<128>(x, d_had_state.signs_pi);
    else if (d == 256) rotate_hadamard<256>(x, d_had_state.signs_pi);

    // 3. MSE quantize
    for (int b = 0; b < TQ3_IDX_BYTES; b++) dst->idx[b] = 0;
    float yt[256];
    for (int j = 0; j < d; j++) {
        int best = 0;
        float bd = fabsf(x[j] - d_had_state.centroids[0]);
        for (int c = 1; c < 4; c++) {
            float dd = fabsf(x[j] - d_had_state.centroids[c]);
            if (dd < bd) { bd = dd; best = c; }
        }
        yt[j] = d_had_state.centroids[best];
        int bp = j * 2;
        dst->idx[bp / 8] |= (uint8_t)((best & 0x3) << (bp % 8));
    }

    // 4. Inverse rotation + residual: O(d log d)
    float x_tilde[256];
    for (int i = 0; i < d; i++) x_tilde[i] = yt[i];
    if (d == 128) inverse_rotate_hadamard<128>(x_tilde, d_had_state.signs_pi);
    else if (d == 256) inverse_rotate_hadamard<256>(x_tilde, d_had_state.signs_pi);

    // x was already rotated, inverse of rotation applied to yt gives us x_tilde in original space
    // residual in original (unrotated) space:
    // Actually we need to un-rotate x first. But we normalized x_unit and rotated it.
    // Let me reconsider: x_unit → rotate → quantize → yt → inverse_rotate → x_tilde
    // residual r = x_unit - x_tilde (in original space)
    // But x was modified in-place by rotate_hadamard...
    // We need to keep a copy of x_unit before rotation.

    // Let me redo with explicit copy:
    float x_unit[256];
    for (int i = 0; i < d; i++) x_unit[i] = src[i] * inv;

    float r[256];
    for (int i = 0; i < d; i++) r[i] = x_unit[i] - x_tilde[i];

    float g2 = 0.0f;
    for (int i = 0; i < d; i++) g2 += r[i] * r[i];

    // 5. QJL: project residual through S (Hadamard)
    float u[256];
    for (int i = 0; i < d; i++) u[i] = r[i];
    if (d == 128) rotate_hadamard<128>(u, d_had_state.signs_s);
    else if (d == 256) rotate_hadamard<256>(u, d_had_state.signs_s);

    for (int b = 0; b < TQ3_SIGN_BYTES; b++) dst->qjl_sign[b] = 0;
    for (int j = 0; j < d; j++) {
        if (u[j] >= 0.0f) dst->qjl_sign[j / 8] |= (uint8_t)(1 << (j % 8));
    }

    dst->x_norm = x_norm;
    dst->gamma = sqrtf(g2);
}

// =====================================================================
// Hadamard dequantize: O(d log d)
// =====================================================================

static __device__ void tq3_dequant_block_hadamard(const block_tq3_0 * src, float * dst) {
    const int d = d_had_state.head_dim;

    // 1. Unpack centroids
    float yt[256];
    for (int j = 0; j < d; j++) {
        int bp = j * 2;
        int idx = (src->idx[bp / 8] >> (bp % 8)) & 0x3;
        yt[j] = d_had_state.centroids[idx];
    }

    // 2. Inverse rotate: O(d log d)
    if (d == 128) inverse_rotate_hadamard<128>(yt, d_had_state.signs_pi);
    else if (d == 256) inverse_rotate_hadamard<256>(yt, d_had_state.signs_pi);

    for (int i = 0; i < d; i++) dst[i] = yt[i];

    // 3. QJL: unpack signs, inverse rotate, scale
    float sign_vec[256];
    for (int j = 0; j < d; j++) {
        sign_vec[j] = (src->qjl_sign[j / 8] >> (j % 8)) & 1 ? 1.0f : -1.0f;
    }
    if (d == 128) inverse_rotate_hadamard<128>(sign_vec, d_had_state.signs_s);
    else if (d == 256) inverse_rotate_hadamard<256>(sign_vec, d_had_state.signs_s);

    float scale = d_had_state.qjl_factor * src->gamma;
    for (int i = 0; i < d; i++) dst[i] += scale * sign_vec[i];

    // 4. Scale by x_norm
    float xn = src->x_norm;
    for (int i = 0; i < d; i++) dst[i] *= xn;
}

// =====================================================================
// Hadamard fused QK: precompute Q rotation O(d log d), then O(d) per KV
// =====================================================================

// For fused attention, the Q precompute becomes:
// Q_rot = rotate_hadamard(Q)  → O(d log d) instead of O(d²)
// Q_proj = rotate_hadamard_s(Q) → O(d log d)
// Then Q·K = x_norm * (dot(Q_rot, centroids[idx]) + qjl * gamma * dot(Q_proj, signs))

// Fused flash attention with Hadamard rotation
template<int D>
__global__ void flash_attn_tq3_hadamard(
    const float * __restrict__ Q,
    const char  * __restrict__ K,
    const char  * __restrict__ V,
    const half  * __restrict__ mask_data,
    float       * __restrict__ dst_data,
    const float scale,
    const int ne01, const int ne02, const int ne11, const int ne12,
    const int k_nb1, const int k_nb2, const int64_t k_nb3,
    const int v_nb1, const int v_nb2, const int64_t v_nb3,
    const int q_nb1, const int q_nb2, const int q_nb3,
    const int mask_nb1
) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int seq = blockIdx.z;
    const int lane = threadIdx.x;

    if (head >= ne02 || token >= ne01) return;

    const int kv_head = ne12 > 0 ? head / (ne02 / ne12) : 0;
    const float * Q_head = (const float *)((const char *)Q + seq*q_nb3 + head*q_nb2 + token*q_nb1);

    extern __shared__ float smem[];
    float * Q_rot  = smem;
    float * Q_proj = smem + D;

    // Precompute Q_rot and Q_proj using HADAMARD: O(d log d) instead of O(d²)!
    // Load Q into shared memory
    for (int i = lane; i < D; i += 32) {
        Q_rot[i] = Q_head[i];
        Q_proj[i] = Q_head[i];
    }
    __syncwarp();

    // Apply Hadamard rotation (single-thread for now — could parallelize butterfly)
    if (lane == 0) {
        rotate_hadamard<D>(Q_rot, d_had_state.signs_pi);
        rotate_hadamard<D>(Q_proj, d_had_state.signs_s);
    }
    __syncwarp();

    // Online softmax attention over KV positions
    float KQ_max = -FLT_MAX;
    float KQ_sum = 0.0f;
    const int ept = (D + 31) / 32;
    float VKQ[8];
    for (int e = 0; e < ept; e++) VKQ[e] = 0.0f;

    for (int kv = 0; kv < ne11; kv++) {
        const block_tq3_0 * K_block = (const block_tq3_0 *)(
            K + seq*k_nb3 + kv_head*(int64_t)k_nb2 + (int64_t)kv*k_nb1);

        // O(d) QK dot product
        float dot_mse = 0.0f, dot_qjl = 0.0f;
        for (int j = lane; j < D; j += 32) {
            int bp = j * 2;
            int idx = (K_block->idx[bp / 8] >> (bp % 8)) & 0x3;
            dot_mse += Q_rot[j] * d_had_state.centroids[idx];
            float sf = (K_block->qjl_sign[j / 8] >> (j % 8)) & 1 ? 1.0f : -1.0f;
            dot_qjl += Q_proj[j] * sf;
        }
        for (int m = 16; m > 0; m >>= 1) {
            dot_mse += __shfl_xor_sync(0xffffffff, dot_mse, m);
            dot_qjl += __shfl_xor_sync(0xffffffff, dot_qjl, m);
        }

        float KQ_val = K_block->x_norm * (dot_mse + d_had_state.qjl_factor * K_block->gamma * dot_qjl) * scale;

        if (mask_data) {
            const half * mr = (const half *)((const char *)mask_data + token * mask_nb1);
            KQ_val += __half2float(mr[kv]);
        }

        float KQ_max_new = fmaxf(KQ_max, KQ_val);
        float corr = expf(KQ_max - KQ_max_new);
        KQ_sum = KQ_sum * corr + expf(KQ_val - KQ_max_new);
        float w = expf(KQ_val - KQ_max_new);

        const half * V_row = (const half *)(V + seq*v_nb3 + kv_head*(int64_t)v_nb2 + (int64_t)kv*v_nb1);
        for (int e = 0; e < ept; e++) {
            int i = lane + e * 32;
            if (i < D) VKQ[e] = VKQ[e] * corr + w * __half2float(V_row[i]);
        }
        KQ_max = KQ_max_new;
    }

    float inv_sum = 1.0f / (KQ_sum + 1e-8f);
    float * out = (float *)((char *)dst_data + seq*ne02*ne01*D*sizeof(float) +
                             head*ne01*D*sizeof(float) + token*D*sizeof(float));
    for (int e = 0; e < ept; e++) {
        int i = lane + e * 32;
        if (i < D) out[i] = VKQ[e] * inv_sum;
    }
}

// =====================================================================
// Device globals (only visible within this TU — no extern needed)
// =====================================================================
static __device__ float * d_pi         = nullptr;
static __device__ float * d_s          = nullptr;
static __device__ float * d_centroids  = nullptr;
static __device__ float   d_qjl_factor = 0.0f;

// Host-side tracking
static float * h_d_pi = nullptr;
static float * h_d_s  = nullptr;
static float * h_d_centroids = nullptr;

// =====================================================================
// Init / free
// =====================================================================

// C-linkage wrappers for cross-library calls
extern "C" void tq3_cuda_init_from_host(const float * pi, const float * s,
                   const float * cent, int k, int d, float qjl) {
    tq3_cuda_init(pi, s, cent, k, d, qjl);
}

extern "C" void tq3_cuda_init_hadamard_from_host(
    const uint8_t * signs_pi, const uint8_t * signs_s,
    const float * cent, int k, int d, float qjl) {
    tq3_cuda_init_hadamard(signs_pi, signs_s, cent, k, d, qjl);
}

void tq3_cuda_init(const float * pi_host, const float * s_host,
                   const float * centroids_host, int k_centroids,
                   int head_dim, float qjl_factor) {
    tq3_cuda_free();

    size_t mat_size = head_dim * head_dim * sizeof(float);
    size_t cent_size = k_centroids * sizeof(float);

    float * dp, * ds, * dc;
    cudaMalloc(&dp, mat_size);
    cudaMalloc(&ds, mat_size);
    cudaMalloc(&dc, cent_size);
    cudaMemcpy(dp, pi_host, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, s_host, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, centroids_host, cent_size, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_pi, &dp, sizeof(float*));
    cudaMemcpyToSymbol(d_s, &ds, sizeof(float*));
    cudaMemcpyToSymbol(d_centroids, &dc, sizeof(float*));
    cudaMemcpyToSymbol(d_qjl_factor, &qjl_factor, sizeof(float));

    h_d_pi = dp; h_d_s = ds; h_d_centroids = dc;

    // Verify device globals are set
    float * check_pi = nullptr;
    float * check_cent = nullptr;
    float check_qjl = 0.0f;
    cudaMemcpyFromSymbol(&check_pi, d_pi, sizeof(float*));
    cudaMemcpyFromSymbol(&check_cent, d_centroids, sizeof(float*));
    cudaMemcpyFromSymbol(&check_qjl, d_qjl_factor, sizeof(float));
    fprintf(stderr, "turboquant CUDA: device state ready (%.1f KB) pi=%p cent=%p qjl=%.6f\n",
            (2 * mat_size + cent_size) / 1024.0f, (void*)check_pi, (void*)check_cent, check_qjl);

    // Quick roundtrip test on GPU
    {
        float h_src[128], h_dst[128];
        for (int i = 0; i < 128; i++) h_src[i] = (i == 0) ? 1.0f : 0.0f; // unit vector e_0
        float * d_src, * d_dst;
        block_tq3_0 * d_packed;
        cudaMalloc(&d_src, 128*sizeof(float));
        cudaMalloc(&d_dst, 128*sizeof(float));
        cudaMalloc(&d_packed, sizeof(block_tq3_0));
        cudaMemcpy(d_src, h_src, 128*sizeof(float), cudaMemcpyHostToDevice);

        // Quantize
        extern __global__ void k_test_quant(const float*, block_tq3_0*);
        extern __global__ void k_test_dequant(const block_tq3_0*, float*);
        k_test_quant<<<1, 1>>>(d_src, d_packed);
        k_test_dequant<<<1, 1>>>(d_packed, d_dst);
        cudaDeviceSynchronize();

        cudaMemcpy(h_dst, d_dst, 128*sizeof(float), cudaMemcpyDeviceToHost);
        float mse = 0;
        for (int i = 0; i < 128; i++) { float d = h_src[i] - h_dst[i]; mse += d*d; }
        fprintf(stderr, "turboquant CUDA: roundtrip test MSE=%.6f dst[0:3]=%.4f %.4f %.4f\n",
                mse/128, h_dst[0], h_dst[1], h_dst[2]);
        cudaFree(d_src); cudaFree(d_dst); cudaFree(d_packed);
    }
}

void tq3_cuda_free() {
    if (h_d_pi) cudaFree(h_d_pi);
    if (h_d_s) cudaFree(h_d_s);
    if (h_d_centroids) cudaFree(h_d_centroids);
    h_d_pi = h_d_s = h_d_centroids = nullptr;
}

// =====================================================================
// Device: quantize one block (128 floats → 56 bytes)
// =====================================================================

static __device__ void tq3_quantize_block(const float * src, block_tq3_0 * dst) {
    const int d = TQ3_HEAD_DIM;
    float x_unit[TQ3_HEAD_DIM], y[TQ3_HEAD_DIM], r[TQ3_HEAD_DIM];

    float x_norm_sq = 0.0f;
    for (int i = 0; i < d; i++) x_norm_sq += src[i] * src[i];
    float x_norm = sqrtf(x_norm_sq);
    float inv = (x_norm > 1e-8f) ? (1.0f / x_norm) : 0.0f;
    for (int i = 0; i < d; i++) x_unit[i] = src[i] * inv;

    // y = x_unit @ Pi^T
    for (int j = 0; j < d; j++) {
        float acc = 0.0f;
        for (int i = 0; i < d; i++) acc += x_unit[i] * d_pi[j * d + i];
        y[j] = acc;
    }

    // MSE quantize
    for (int b = 0; b < TQ3_IDX_BYTES; b++) dst->idx[b] = 0;
    float yt[TQ3_HEAD_DIM];
    for (int j = 0; j < d; j++) {
        int best = 0;
        float bd = fabsf(y[j] - d_centroids[0]);
        for (int c = 1; c < TQ3_K_CENTROIDS; c++) {
            float dd = fabsf(y[j] - d_centroids[c]);
            if (dd < bd) { bd = dd; best = c; }
        }
        yt[j] = d_centroids[best];
        int bp = j * 2;
        dst->idx[bp / 8] |= (uint8_t)((best & 0x3) << (bp % 8));
    }

    // Inverse rotate + residual
    for (int i = 0; i < d; i++) {
        float acc = 0.0f;
        for (int j = 0; j < d; j++) acc += yt[j] * d_pi[j * d + i];
        r[i] = x_unit[i] - acc;
    }
    float g2 = 0.0f;
    for (int i = 0; i < d; i++) g2 += r[i] * r[i];

    // QJL signs
    for (int b = 0; b < TQ3_SIGN_BYTES; b++) dst->qjl_sign[b] = 0;
    for (int j = 0; j < d; j++) {
        float acc = 0.0f;
        for (int i = 0; i < d; i++) acc += r[i] * d_s[j * d + i];
        if (acc >= 0.0f) dst->qjl_sign[j / 8] |= (uint8_t)(1 << (j % 8));
    }

    dst->x_norm = x_norm;
    dst->gamma = sqrtf(g2);
}

// =====================================================================
// Device: dequantize one block (56 bytes → 128 floats)
// =====================================================================

static __device__ void tq3_dequant_block(const block_tq3_0 * src, float * dst) {
    const int d = TQ3_HEAD_DIM;
    float yt[TQ3_HEAD_DIM];

    for (int j = 0; j < d; j++) {
        int bp = j * 2;
        int idx = (src->idx[bp / 8] >> (bp % 8)) & 0x3;
        yt[j] = d_centroids[idx];
    }

    for (int i = 0; i < d; i++) {
        float acc = 0.0f;
        for (int j = 0; j < d; j++) acc += yt[j] * d_pi[j * d + i];
        dst[i] = acc;
    }

    float scale = d_qjl_factor * src->gamma;
    for (int i = 0; i < d; i++) {
        float acc = 0.0f;
        for (int j = 0; j < d; j++) {
            float sf = (src->qjl_sign[j / 8] >> (j % 8)) & 1 ? 1.0f : -1.0f;
            acc += sf * d_s[j * d + i];
        }
        dst[i] += scale * acc;
    }

    float xn = src->x_norm;
    for (int i = 0; i < d; i++) dst[i] *= xn;
}

// =====================================================================
// Test kernels for roundtrip verification
// =====================================================================

__global__ void k_test_quant(const float * src, block_tq3_0 * dst) {
    tq3_quantize_block(src, dst);
}

__global__ void k_test_dequant(const block_tq3_0 * src, float * dst) {
    tq3_dequant_block(src, dst);
}

// =====================================================================
// Set-rows kernel
// =====================================================================

template <typename idx_t>
static __global__ void k_tq3_set_rows(
    const float * __restrict__ src0,
    const idx_t * __restrict__ src1,
    block_tq3_0 * __restrict__ dst,
    int64_t ne00, int64_t ne01, int64_t s01, int64_t s1
) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t n_heads = ne00 / TQ3_HEAD_DIM;
    const int64_t total = ne01 * n_heads;
    if (i >= total) return;

    const int64_t ih = i % n_heads;
    const int64_t ir = i / n_heads;

    const float * sb = src0 + ir * s01 + ih * TQ3_HEAD_DIM;
    const int64_t dr = src1[ir];
    block_tq3_0 * db = (block_tq3_0 *)((char *)dst + dr * s1) + ih;

    if (d_had_state.initialized) {
        tq3_quantize_block_hadamard(sb, db);
    } else {
        tq3_quantize_block(sb, db);
    }
}

static int g_set_rows_calls = 0;
void tq3_cuda_set_rows(
    const float * src0_d, const void * src1_d, void * dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    bool idx64, cudaStream_t stream
) {
    if (g_set_rows_calls < 3) {
        fprintf(stderr, "TQ3 CUDA set_rows: ne00=%lld ne01=%lld nb01=%zu nb1=%zu idx64=%d\n",
                (long long)ne00, (long long)ne01, nb01, nb1, (int)idx64);
    }
    g_set_rows_calls++;
    const int64_t n_heads = ne00 / TQ3_HEAD_DIM;
    const int64_t total = ne01 * n_heads;
    if (total <= 0) return;

    const int bs = 32;
    const int gs = (total + bs - 1) / bs;
    const int64_t s01 = nb01 / sizeof(float);

    if (idx64) {
        k_tq3_set_rows<int64_t><<<gs, bs, 0, stream>>>(
            src0_d, (const int64_t *)src1_d, (block_tq3_0 *)dst_d,
            ne00, ne01, s01, nb1);
    } else {
        k_tq3_set_rows<int32_t><<<gs, bs, 0, stream>>>(
            src0_d, (const int32_t *)src1_d, (block_tq3_0 *)dst_d,
            ne00, ne01, s01, nb1);
    }
}

// =====================================================================
// Dequantize kernel (for cpy/dup TQ3_0 → F32)
// =====================================================================

// =====================================================================
// Strided copy TQ3_0 → F32 (proper view handling)
// =====================================================================

static __global__ void k_tq3_cpy_to_f32(
    const char * __restrict__ src, char * __restrict__ dst,
    int64_t ne, int64_t ne00, int64_t ne01, int64_t ne02,
    size_t nb00, size_t nb01, size_t nb02, size_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb10, size_t nb11, size_t nb12, size_t nb13
) {
    // Each thread handles one TQ3_0 block = 128 elements
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t n_blocks_per_row = ne00 / TQ3_HEAD_DIM;
    const int64_t total_blocks = n_blocks_per_row * ne01 * ne02;

    if (i >= total_blocks) return;

    // Decompose flat index into (block_in_row, row, layer)
    const int64_t ib = i % n_blocks_per_row;
    const int64_t i01 = (i / n_blocks_per_row) % ne01;
    const int64_t i02 = i / (n_blocks_per_row * ne01);

    // Source: TQ3_0 data with strides
    // nb00 = type_size / blck_size * blck_size = 56 (for one block)
    // nb01 = stride between rows in bytes
    const block_tq3_0 * src_block = (const block_tq3_0 *)(src + i02*nb02 + i01*nb01) + ib;

    // Destination: contiguous F32
    float * dst_row = (float *)(dst + i02*nb12 + i01*nb11) + ib * TQ3_HEAD_DIM;

    float tmp[TQ3_HEAD_DIM];
    tq3_dequant_block(src_block, tmp);
    for (int j = 0; j < TQ3_HEAD_DIM; j++) {
        dst_row[j] = tmp[j];
    }
}

void tq3_cuda_cpy_to_f32(
    const char * src, char * dst, int64_t ne,
    int64_t ne00, int64_t ne01, int64_t ne02,
    size_t nb00, size_t nb01, size_t nb02, size_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb10, size_t nb11, size_t nb12, size_t nb13,
    cudaStream_t stream
) {
    const int64_t n_blocks_per_row = ne00 / TQ3_HEAD_DIM;
    const int64_t total_blocks = n_blocks_per_row * ne01 * ne02;
    if (total_blocks <= 0) return;

    const int bs = 32;
    const int gs = (total_blocks + bs - 1) / bs;

    k_tq3_cpy_to_f32<<<gs, bs, 0, stream>>>(
        src, dst, ne, ne00, ne01, ne02,
        nb00, nb01, nb02, nb03,
        ne10, ne11, ne12,
        nb10, nb11, nb12, nb13);
}

// =====================================================================
// Strided copy TQ3_0 → F16 (direct, skips F32 intermediate)
// =====================================================================

// Optimized v3: precompute centroid vector and sign vector, then matmul
// Eliminates redundant index lookups in the inner loop
static __global__ void k_tq3_cpy_to_f16(
    const char * __restrict__ src, char * __restrict__ dst,
    int64_t ne, int64_t ne00, int64_t ne01, int64_t ne02,
    size_t nb00, size_t nb01, size_t nb02, size_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb10, size_t nb11, size_t nb12, size_t nb13
) {
    // Block layout: blockDim.x=32 (warp), blockDim.y=warps_per_block
    const int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;

    const int d = (int)ne00;
    const int64_t total_blocks = ne01 * ne02;
    if (warp_id >= total_blocks) return;

    const int64_t i01 = warp_id % ne01;
    const int64_t i02 = warp_id / ne01;

    const block_tq3_0 * sb = (const block_tq3_0 *)(src + i02*nb02 + i01*nb01);
    half * dr = (half *)(dst + i02*nb12 + i01*nb11);

    const float xn = sb->x_norm;
    const float gm = sb->gamma;
    const float qf = d_qjl_factor;

    // Shared memory: precomputed y_tilde (centroid values) and sign_f
    // These are the same for all output elements within one block
    extern __shared__ float smem[];
    const int smem_offset = threadIdx.y * d * 2;
    float * y_tilde = smem + smem_offset;
    float * sign_f  = smem + smem_offset + d;

    // Step 1: Cooperatively decode idx→centroid and sign bits (O(d/32) per thread)
    for (int j = lane; j < d; j += 32) {
        int bp = j * 2;
        int idx = (sb->idx[bp / 8] >> (bp % 8)) & 0x3;
        y_tilde[j] = d_centroids[idx];
        sign_f[j] = (sb->qjl_sign[j / 8] >> (j % 8)) & 1 ? 1.0f : -1.0f;
    }
    __syncwarp();

    // Step 2: Matmul — each lane handles d/32 output elements
    // dst[i] = xn * (sum_j y_tilde[j]*Pi[j*d+i] + qf*gm * sum_j sign_f[j]*S[j*d+i])
    for (int i = lane; i < d; i += 32) {
        float acc_mse = 0.0f;
        float acc_qjl = 0.0f;
        for (int j = 0; j < d; j++) {
            acc_mse += y_tilde[j] * __ldg(&d_pi[j * d + i]);
            acc_qjl += sign_f[j]  * __ldg(&d_s[j * d + i]);
        }
        dr[i] = __float2half(xn * (acc_mse + qf * gm * acc_qjl));
    }
}

void tq3_cuda_cpy_to_f16(
    const char * src, char * dst, int64_t ne,
    int64_t ne00, int64_t ne01, int64_t ne02,
    size_t nb00, size_t nb01, size_t nb02, size_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb10, size_t nb11, size_t nb12, size_t nb13,
    cudaStream_t stream
) {
    const int64_t head_dim = ne00;
    const int64_t n_blocks_per_row = 1;  // head-dim view
    const int64_t total_blocks = n_blocks_per_row * ne01 * ne02;
    if (total_blocks <= 0) return;

    // Use warp-based launch: 32 threads per warp, N warps per CUDA block
    // Each warp needs 2*d floats of shared memory for y_tilde + sign_f
    const int d = (int)ne00;
    const int smem_per_warp = 2 * d * sizeof(float);
    const int max_smem = 48 * 1024;  // 48KB typical
    const int warps_per_block = max_smem / smem_per_warp;
    const int wpb = warps_per_block > 4 ? 4 : (warps_per_block > 0 ? warps_per_block : 1);
    dim3 block(32, wpb);
    dim3 grid((total_blocks + wpb - 1) / wpb);
    const int smem_size = wpb * smem_per_warp;

    k_tq3_cpy_to_f16<<<grid, block, smem_size, stream>>>(
        src, dst, ne, ne00, ne01, ne02,
        nb00, nb01, nb02, nb03,
        ne10, ne11, ne12,
        nb10, nb11, nb12, nb13);
}

// =====================================================================
// Flat dequantize (kept for reference)
// =====================================================================

static __global__ void k_dequant_tq3(const block_tq3_0 * __restrict__ src,
                                      float * __restrict__ dst, int64_t n) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    tq3_dequant_block(src + i, dst + i * TQ3_HEAD_DIM);
}

void dequantize_row_tq3_0_cuda(const void * src, float * dst, int64_t k, cudaStream_t stream) {
    const int64_t n = k / TQ3_HEAD_DIM;
    if (n <= 0) return;
    k_dequant_tq3<<<(n + 31) / 32, 32, 0, stream>>>((const block_tq3_0 *)src, dst, n);
}

// =====================================================================
// Fused flash attention for TQ3_0 K (moved from fattn-tq3.cu)
// Must be in same TU as device globals d_pi, d_s, d_centroids
// =====================================================================

#include <cfloat>

template<int D>
__global__ void flash_attn_tq3_vec_impl(
    const float * __restrict__ Q,
    const char  * __restrict__ K,
    const char  * __restrict__ V,
    const half  * __restrict__ mask_data,
    float       * __restrict__ dst_data,
    const float scale,
    const int ne01, const int ne02, const int ne11, const int ne12,
    const int k_nb1, const int k_nb2, const int64_t k_nb3,
    const int v_nb1, const int v_nb2, const int64_t v_nb3,
    const int q_nb1, const int q_nb2, const int q_nb3,
    const int mask_nb1
) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int seq = blockIdx.z;
    const int lane = threadIdx.x;

    if (head >= ne02 || token >= ne01) return;

    const int kv_head = ne12 > 0 ? head / (ne02 / ne12) : 0;
    const float * Q_head = (const float *)((const char *)Q + seq*q_nb3 + head*q_nb2 + token*q_nb1);

    extern __shared__ float smem[];
    float * Q_rot  = smem;
    float * Q_proj = smem + D;

    // Precompute Q_rot and Q_proj
    for (int j = lane; j < D; j += 32) {
        float acc_pi = 0.0f, acc_s = 0.0f;
        for (int i = 0; i < D; i++) {
            float qi = Q_head[i];
            acc_pi += qi * d_pi[j * D + i];
            acc_s  += qi * d_s[j * D + i];
        }
        Q_rot[j] = acc_pi;
        Q_proj[j] = acc_s;
    }
    __syncwarp();

    float KQ_max = -FLT_MAX;
    float KQ_sum = 0.0f;
    const int ept = (D + 31) / 32;
    float VKQ[8]; // max ept
    for (int e = 0; e < ept; e++) VKQ[e] = 0.0f;

    for (int kv = 0; kv < ne11; kv++) {
        const block_tq3_0 * K_block = (const block_tq3_0 *)(
            K + seq*k_nb3 + kv_head*(int64_t)k_nb2 + (int64_t)kv*k_nb1);

        float dot_mse = 0.0f, dot_qjl = 0.0f;
        for (int j = lane; j < D; j += 32) {
            int bp = j * 2;
            int idx = (K_block->idx[bp / 8] >> (bp % 8)) & 0x3;
            dot_mse += Q_rot[j] * d_centroids[idx];
            float sf = (K_block->qjl_sign[j / 8] >> (j % 8)) & 1 ? 1.0f : -1.0f;
            dot_qjl += Q_proj[j] * sf;
        }
        for (int m = 16; m > 0; m >>= 1) {
            dot_mse += __shfl_xor_sync(0xffffffff, dot_mse, m);
            dot_qjl += __shfl_xor_sync(0xffffffff, dot_qjl, m);
        }

        float KQ_val = K_block->x_norm * (dot_mse + d_qjl_factor * K_block->gamma * dot_qjl) * scale;

        if (mask_data) {
            const half * mr = (const half *)((const char *)mask_data + token * mask_nb1);
            KQ_val += __half2float(mr[kv]);
        }

        float KQ_max_new = fmaxf(KQ_max, KQ_val);
        float corr = expf(KQ_max - KQ_max_new);
        KQ_sum = KQ_sum * corr + expf(KQ_val - KQ_max_new);
        float w = expf(KQ_val - KQ_max_new);

        const half * V_row = (const half *)(V + seq*v_nb3 + kv_head*(int64_t)v_nb2 + (int64_t)kv*v_nb1);
        for (int e = 0; e < ept; e++) {
            int i = lane + e * 32;
            if (i < D) VKQ[e] = VKQ[e] * corr + w * __half2float(V_row[i]);
        }
        KQ_max = KQ_max_new;
    }

    float inv_sum = 1.0f / (KQ_sum + 1e-8f);
    float * out = (float *)((char *)dst_data + seq*ne02*ne01*D*sizeof(float) +
                             head*ne01*D*sizeof(float) + token*D*sizeof(float));
    for (int e = 0; e < ept; e++) {
        int i = lane + e * 32;
        if (i < D) out[i] = VKQ[e] * inv_sum;
    }
}

bool ggml_cuda_flash_attn_ext_tq3(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (K->type != GGML_TYPE_TQ3_0) return false;
    if (V->type != GGML_TYPE_F16) return false;

    const int D = Q->ne[0];
    if (D != 128 && D != 256) return false;

    float scale = 1.0f, max_bias = 0.0f;
    memcpy(&scale, (const float *)dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *)dst->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) return false;

    const int ne01 = Q->ne[1], ne02 = Q->ne[2], ne03 = Q->ne[3];
    const int ne11 = K->ne[1], ne12 = K->ne[2];
    const int smem = 2 * D * sizeof(float);

    dim3 grid(ne02, ne01, ne03);
    dim3 block(32);
    cudaStream_t stream = ctx.stream();

    const int mask_nb1 = mask ? mask->nb[1] : 0;

    // Dispatch: Hadamard mode (O(d log d) precompute) vs Dense mode (O(d²))
    if (h_had_initialized) {
        if (D == 128) {
            flash_attn_tq3_hadamard<128><<<grid, block, smem, stream>>>(
                (const float *)Q->data, (const char *)K->data, (const char *)V->data,
                mask ? (const half *)mask->data : nullptr,
                (float *)dst->data, scale,
                ne01, ne02, ne11, ne12,
                K->nb[1], K->nb[2], K->nb[3],
                V->nb[1], V->nb[2], V->nb[3],
                Q->nb[1], Q->nb[2], Q->nb[3],
                mask_nb1);
        } else if (D == 256) {
            flash_attn_tq3_hadamard<256><<<grid, block, smem, stream>>>(
                (const float *)Q->data, (const char *)K->data, (const char *)V->data,
                mask ? (const half *)mask->data : nullptr,
                (float *)dst->data, scale,
                ne01, ne02, ne11, ne12,
                K->nb[1], K->nb[2], K->nb[3],
                V->nb[1], V->nb[2], V->nb[3],
                Q->nb[1], Q->nb[2], Q->nb[3],
                mask_nb1);
        }
        return true;
    }

    // Dense mode fallback
    if (D == 128) {
        flash_attn_tq3_vec_impl<128><<<grid, block, smem, stream>>>(
            (const float *)Q->data, (const char *)K->data, (const char *)V->data,
            mask ? (const half *)mask->data : nullptr,
            (float *)dst->data, scale,
            ne01, ne02, ne11, ne12,
            K->nb[1], K->nb[2], K->nb[3],
            V->nb[1], V->nb[2], V->nb[3],
            Q->nb[1], Q->nb[2], Q->nb[3],
            mask_nb1);
    } else if (D == 256) {
        flash_attn_tq3_vec_impl<256><<<grid, block, smem, stream>>>(
            (const float *)Q->data, (const char *)K->data, (const char *)V->data,
            mask ? (const half *)mask->data : nullptr,
            (float *)dst->data, scale,
            ne01, ne02, ne11, ne12,
            K->nb[1], K->nb[2], K->nb[3],
            V->nb[1], V->nb[2], V->nb[3],
            Q->nb[1], Q->nb[2], Q->nb[3],
            mask_nb1);
    }
    return true;
}
