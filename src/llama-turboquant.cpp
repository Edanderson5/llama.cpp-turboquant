// SPDX-License-Identifier: MIT
// TurboQuant KV cache compression — Phase 1 baseline (CPU)
//
// Quantize: rotation by Pi, scalar MSE quantization, QJL sign residual
// Dequantize: reconstruct from indices + norms + signs + gamma
//
// Reference: Braun et al. 2024, "Identifying Functionally Important Features
// with End-to-End Sparse Dictionary Learning" (ICLR 2026)

#include "llama-turboquant.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace turboquant {

// -------------------------------------------------------------------------
// Sidecar loading (adapted from turboquant_sidecar_loader.cpp)
// -------------------------------------------------------------------------

static constexpr char kMagic[8] = {'T','U','R','B','O','Q','T','1'};

static void read_u32(const uint8_t * p, uint32_t & o) {
    std::memcpy(&o, p, 4);
}

static void read_f64(const uint8_t * p, double & o) {
    std::memcpy(&o, p, 8);
}

meta load_meta(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        throw std::runtime_error("turboquant: cannot open " + path);
    }
    auto size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> blob(size);
    f.read(reinterpret_cast<char *>(blob.data()), size);

    if (blob.size() < 32) {
        throw std::runtime_error("turboquant: truncated header");
    }
    if (std::memcmp(blob.data(), kMagic, 8) != 0) {
        throw std::runtime_error("turboquant: bad magic in " + path);
    }

    uint32_t ver = 0;
    read_u32(blob.data() + 8, ver);

    meta m;
    size_t hlen = 0;
    uint32_t d = 0, k = 0;

    if (ver == 1) {
        uint32_t bits_i = 0;
        read_u32(blob.data() + 12, bits_i);
        m.bits = double(bits_i);
        read_u32(blob.data() + 16, d);
        read_u32(blob.data() + 20, k);
        read_f64(blob.data() + 24, m.qjl_factor);
        m.codebook_id = 0;
        hlen = 32;
    } else if (ver == 2) {
        read_f64(blob.data() + 12, m.bits);
        read_u32(blob.data() + 20, d);
        read_u32(blob.data() + 24, k);
        uint32_t cbid = 0;
        read_u32(blob.data() + 28, cbid);
        m.codebook_id = int(cbid);
        read_f64(blob.data() + 32, m.qjl_factor);
        hlen = 40;
    } else {
        throw std::runtime_error("turboquant: unsupported version " + std::to_string(ver));
    }

    m.head_dim     = int(d);
    m.k_centroids  = int(k);

    size_t need = hlen + size_t(k) * 4u + 2u * size_t(d) * size_t(d) * 4u;
    if (blob.size() < need) {
        throw std::runtime_error("turboquant: truncated payload");
    }

    m.centroids.resize(k);
    m.pi.resize(size_t(d) * size_t(d));
    m.s.resize(size_t(d) * size_t(d));

    size_t off = hlen;
    std::memcpy(m.centroids.data(), blob.data() + off, k * 4u);
    off += k * 4u;
    std::memcpy(m.pi.data(), blob.data() + off, size_t(d) * size_t(d) * 4u);
    off += size_t(d) * size_t(d) * 4u;
    std::memcpy(m.s.data(), blob.data() + off, size_t(d) * size_t(d) * 4u);

    double expected_qjl = std::sqrt(M_PI / 2.0) / double(d);
    if (std::abs(m.qjl_factor - expected_qjl) > 1e-5) {
        throw std::runtime_error("turboquant: qjl_factor mismatch");
    }

    return m;
}

// -------------------------------------------------------------------------
// Row layout computation
// -------------------------------------------------------------------------

row_layout compute_row_layout(const meta & m) {
    row_layout l;

    // bits per MSE centroid index
    int bits_per_idx = 0;
    int k = m.k_centroids;
    while ((1 << bits_per_idx) < k) {
        bits_per_idx++;
    }
    l.idx_bits_per_coord = bits_per_idx;
    l.idx_bytes   = (uint32_t)((m.head_dim * bits_per_idx + 7) / 8);
    l.sign_bytes  = (uint32_t)((m.head_dim + 7) / 8);
    l.norm_bytes  = 4;  // float32
    l.gamma_bytes = 4;  // float32

    // Layout: [idx | norm | sign | gamma]
    l.norm_offset  = l.idx_bytes;
    l.sign_offset  = l.norm_offset + l.norm_bytes;
    l.gamma_offset = l.sign_offset + l.sign_bytes;
    l.total_bytes  = l.gamma_offset + l.gamma_bytes;

    return l;
}

// -------------------------------------------------------------------------
// Bit packing helpers
// -------------------------------------------------------------------------

static void pack_indices(
    const int * idx, int n, int bits_per_idx, uint8_t * dst
) {
    int total_bits = n * bits_per_idx;
    int total_bytes = (total_bits + 7) / 8;
    std::memset(dst, 0, total_bytes);

    for (int i = 0; i < n; i++) {
        uint32_t val = (uint32_t)idx[i];
        int bit_pos = i * bits_per_idx;
        for (int b = 0; b < bits_per_idx; b++) {
            if (val & (1u << b)) {
                int pos = bit_pos + b;
                dst[pos / 8] |= (1u << (pos % 8));
            }
        }
    }
}

static void unpack_indices(
    const uint8_t * src, int n, int bits_per_idx, int * idx
) {
    uint32_t mask = (1u << bits_per_idx) - 1u;
    for (int i = 0; i < n; i++) {
        uint32_t val = 0;
        int bit_pos = i * bits_per_idx;
        for (int b = 0; b < bits_per_idx; b++) {
            int pos = bit_pos + b;
            if (src[pos / 8] & (1u << (pos % 8))) {
                val |= (1u << b);
            }
        }
        idx[i] = int(val & mask);
    }
}

static void pack_signs(const uint8_t * signs, int n, uint8_t * dst) {
    int nbytes = (n + 7) / 8;
    std::memset(dst, 0, nbytes);
    for (int i = 0; i < n; i++) {
        if (signs[i]) {
            dst[i / 8] |= (1u << (i % 8));
        }
    }
}

static void unpack_signs(const uint8_t * src, int n, float * sign_f) {
    for (int i = 0; i < n; i++) {
        bool bit = (src[i / 8] >> (i % 8)) & 1u;
        sign_f[i] = bit ? 1.0f : -1.0f;
    }
}

// -------------------------------------------------------------------------
// Quantize one head-row
// -------------------------------------------------------------------------

void quantize_head_row(
    const meta       & m,
    const row_layout & layout,
    const float      * src,
    uint8_t          * dst
) {
    const int d = m.head_dim;
    const int k = m.k_centroids;

    // Temporary buffers (small, stack-friendly for d <= 256)
    std::vector<float> x_unit(d);
    std::vector<float> y(d);
    std::vector<float> y_tilde(d);
    std::vector<float> x_tilde(d);
    std::vector<float> r(d);
    std::vector<float> u(d);
    std::vector<int>   idx(d);
    std::vector<uint8_t> signs(d);  // use uint8_t instead of bool (vector<bool> has no .data())

    // 1. Normalize: x_norm, x_unit
    float x_norm = 0.0f;
    for (int i = 0; i < d; i++) {
        x_norm += src[i] * src[i];
    }
    x_norm = std::sqrt(x_norm);
    float inv_norm = (x_norm > 1e-8f) ? (1.0f / x_norm) : 0.0f;
    for (int i = 0; i < d; i++) {
        x_unit[i] = src[i] * inv_norm;
    }

    // 2. Rotate: y = x_unit @ Pi^T  (Pi is row-major [d,d])
    // y[j] = sum_i x_unit[i] * Pi[i][j] = sum_i x_unit[i] * pi[i*d + j]
    for (int j = 0; j < d; j++) {
        float s = 0.0f;
        for (int i = 0; i < d; i++) {
            s += x_unit[i] * m.pi[i * d + j];
        }
        y[j] = s;
    }

    // 3. Scalar MSE quantization: find nearest centroid for each coordinate
    for (int j = 0; j < d; j++) {
        int best = 0;
        float best_dist = std::abs(y[j] - m.centroids[0]);
        for (int c = 1; c < k; c++) {
            float dist = std::abs(y[j] - m.centroids[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best = c;
            }
        }
        idx[j] = best;
        y_tilde[j] = m.centroids[best];
    }

    // 4. Inverse rotate: x_tilde = y_tilde @ Pi  (row vec times matrix)
    // x_tilde[i] = sum_j y_tilde[j] * Pi[j][i] = sum_j y_tilde[j] * pi[j*d + i]
    for (int i = 0; i < d; i++) {
        float s = 0.0f;
        for (int j = 0; j < d; j++) {
            s += y_tilde[j] * m.pi[j * d + i];
        }
        x_tilde[i] = s;
    }

    // 5. Residual
    for (int i = 0; i < d; i++) {
        r[i] = x_unit[i] - x_tilde[i];
    }
    float gamma = 0.0f;
    for (int i = 0; i < d; i++) {
        gamma += r[i] * r[i];
    }
    gamma = std::sqrt(gamma);

    // 6. QJL: u = r @ S^T, sign = (u >= 0)
    // u[j] = sum_i r[i] * S[i][j] = sum_i r[i] * s[i*d + j]
    for (int j = 0; j < d; j++) {
        float s = 0.0f;
        for (int i = 0; i < d; i++) {
            s += r[i] * m.s[i * d + j];
        }
        signs[j] = (s >= 0.0f);
    }

    // 7. Pack into dst
    pack_indices(idx.data(), d, layout.idx_bits_per_coord, dst);
    std::memcpy(dst + layout.norm_offset, &x_norm, 4);
    pack_signs(signs.data(), d, dst + layout.sign_offset);
    std::memcpy(dst + layout.gamma_offset, &gamma, 4);
}

// -------------------------------------------------------------------------
// Dequantize one head-row
// -------------------------------------------------------------------------

void dequant_head_row(
    const meta       & m,
    const row_layout & layout,
    const uint8_t    * src,
    float            * dst
) {
    const int d = m.head_dim;

    std::vector<int>   idx(d);
    std::vector<float> sign_f(d);
    std::vector<float> y_tilde(d);

    // Unpack
    unpack_indices(src, d, layout.idx_bits_per_coord, idx.data());
    float x_norm = 0.0f;
    std::memcpy(&x_norm, src + layout.norm_offset, 4);
    unpack_signs(src + layout.sign_offset, d, sign_f.data());
    float gamma = 0.0f;
    std::memcpy(&gamma, src + layout.gamma_offset, 4);

    // Centroid lookup
    for (int j = 0; j < d; j++) {
        int t = idx[j];
        assert(t >= 0 && t < m.k_centroids);
        y_tilde[j] = m.centroids[t];
    }

    // x_mse[i] = sum_j y_tilde[j] * Pi[j][i]
    for (int i = 0; i < d; i++) {
        float s = 0.0f;
        for (int j = 0; j < d; j++) {
            s += y_tilde[j] * m.pi[j * d + i];
        }
        dst[i] = s;
    }

    // x_qjl[i] = qjl_factor * gamma * sum_j sign_f[j] * S[j][i]
    float scale = float(m.qjl_factor) * gamma;
    for (int i = 0; i < d; i++) {
        float acc = 0.0f;
        for (int j = 0; j < d; j++) {
            acc += sign_f[j] * m.s[j * d + i];
        }
        dst[i] += scale * acc;
    }

    // Scale by x_norm
    for (int i = 0; i < d; i++) {
        dst[i] *= x_norm;
    }
}

// -------------------------------------------------------------------------
// Post-process: quantize + dequantize one KV row in-place
// -------------------------------------------------------------------------

void post_process_row(
    state        & st,
    ggml_tensor  * k_tensor,
    ggml_tensor  * v_tensor,
    uint32_t       row_idx,
    uint32_t       layer_idx,
    int            n_kv_heads,
    int            head_dim
) {
    if (!st.enabled || layer_idx >= st.layers.size()) return;

    auto & lb = st.layers[layer_idx];
    const auto & m = st.m;
    const auto & l = st.layout;
    const size_t row_stride = size_t(n_kv_heads) * l.total_bytes;

    // Process K
    {
        // Read one row from K cache: [n_kv_heads * head_dim] floats
        // K tensor shape is [n_embd_k_gqa, kv_size, ...]
        // For row_idx, we read n_embd_k_gqa = n_kv_heads * head_dim values
        const int n_embd = n_kv_heads * head_dim;
        std::vector<float> row_f32(n_embd);

        // Read from tensor at row_idx
        size_t offset = size_t(row_idx) * n_embd * ggml_type_size(k_tensor->type);
        ggml_backend_tensor_get(k_tensor, row_f32.data(), offset,
                                n_embd * sizeof(float));

        // Quantize each head, then dequantize back
        uint8_t * packed_row = lb.k_packed.data() + size_t(row_idx) * row_stride;
        std::vector<float> recon(n_embd);

        for (int h = 0; h < n_kv_heads; h++) {
            float * head_src = row_f32.data() + h * head_dim;
            uint8_t * head_packed = packed_row + h * l.total_bytes;
            float * head_dst = recon.data() + h * head_dim;

            quantize_head_row(m, l, head_src, head_packed);
            dequant_head_row(m, l, head_packed, head_dst);
        }

        // Write reconstructed values back
        ggml_backend_tensor_set(k_tensor, recon.data(), offset,
                                n_embd * sizeof(float));
    }

    // Process V (same layout when flash_attn is enabled)
    {
        const int n_embd = n_kv_heads * head_dim;
        std::vector<float> row_f32(n_embd);

        size_t offset = size_t(row_idx) * n_embd * ggml_type_size(v_tensor->type);
        ggml_backend_tensor_get(v_tensor, row_f32.data(), offset,
                                n_embd * sizeof(float));

        uint8_t * packed_row = lb.v_packed.data() + size_t(row_idx) * row_stride;
        std::vector<float> recon(n_embd);

        for (int h = 0; h < n_kv_heads; h++) {
            float * head_src = row_f32.data() + h * head_dim;
            uint8_t * head_packed = packed_row + h * l.total_bytes;
            float * head_dst = recon.data() + h * head_dim;

            quantize_head_row(m, l, head_src, head_packed);
            dequant_head_row(m, l, head_packed, head_dst);
        }

        ggml_backend_tensor_set(v_tensor, recon.data(), offset,
                                n_embd * sizeof(float));
    }
}

// -------------------------------------------------------------------------
// Initialize full TurboQuant state
// -------------------------------------------------------------------------

std::unique_ptr<state> init(
    const std::string & tqmeta_path,
    uint32_t kv_size,
    int n_layers,
    int n_kv_heads,
    int head_dim
) {
    auto st = std::make_unique<state>();

    if (tqmeta_path.empty()) {
        st->enabled = false;
        return st;
    }

    st->m = load_meta(tqmeta_path);

    if (st->m.head_dim != head_dim) {
        throw std::runtime_error(
            "turboquant: sidecar head_dim=" + std::to_string(st->m.head_dim) +
            " != model head_dim=" + std::to_string(head_dim));
    }

    st->layout     = compute_row_layout(st->m);
    st->n_kv_heads = n_kv_heads;
    st->head_dim   = head_dim;
    st->kv_size    = kv_size;

    size_t row_bytes = size_t(n_kv_heads) * st->layout.total_bytes;
    size_t total_bytes = size_t(kv_size) * row_bytes;

    st->layers.resize(n_layers);
    for (auto & lb : st->layers) {
        lb.k_packed.resize(total_bytes, 0);
        lb.v_packed.resize(total_bytes, 0);
    }

    st->enabled = true;

    fprintf(stderr, "turboquant: initialized — bits=%.1f head_dim=%d k_centroids=%d "
            "packed_row=%u bytes/head, shadow=%.1f MB/layer\n",
            st->m.bits, head_dim, st->m.k_centroids,
            st->layout.total_bytes,
            double(total_bytes) / (1024.0 * 1024.0));

    return st;
}

} // namespace turboquant
