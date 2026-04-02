// TurboQuant CUDA — all device code in one TU for __device__ global visibility
#include "turboquant.cuh"
#include <cstdio>
#include <cmath>

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

// C-linkage wrapper for cross-library calls
extern "C" void tq3_cuda_init_from_host(const float * pi, const float * s,
                   const float * cent, int k, int d, float qjl) {
    tq3_cuda_init(pi, s, cent, k, d, qjl);
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

    fprintf(stderr, "turboquant CUDA: device state ready (%.1f KB)\n",
            (2 * mat_size + cent_size) / 1024.0f);
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

    tq3_quantize_block(sb, db);
}

void tq3_cuda_set_rows(
    const float * src0_d, const void * src1_d, void * dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    bool idx64, cudaStream_t stream
) {
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
