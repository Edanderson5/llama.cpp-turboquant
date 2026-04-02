// Standalone TurboQuant roundtrip test
// Build: g++ -std=c++17 -I../src -I../ggml/include -I../include tests/test-turboquant.cpp src/llama-turboquant.cpp -lm -o test-tq
// (doesn't need ggml libs — we just test the core quant/dequant)

#include "llama-turboquant.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Stub the ggml functions we reference but don't need for this test
extern "C" {
    struct ggml_tensor;
    size_t ggml_type_size(int) { return 0; }
    void ggml_backend_tensor_get(ggml_tensor*, void*, size_t, size_t) {}
    void ggml_backend_tensor_set(ggml_tensor*, const void*, size_t, size_t) {}
    struct ggml_type_traits { const char* type_name; };
    void ggml_set_type_traits_funcs(int, void*, void*) {}
    void ggml_set_type_traits_cpu_from_float(int, void*) {}
}

static float randf() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

int main() {
    printf("=== TurboQuant Roundtrip Test ===\n\n");

    // Create a synthetic .tqmeta-like state
    const int head_dim = 128;
    const int bits = 3;

    // Generate random Pi (orthogonal) and S matrices
    // For testing, just use random matrices — not orthogonal but sufficient for roundtrip check
    turboquant::meta m;
    m.bits = bits;
    m.head_dim = head_dim;
    m.qjl_factor = sqrt(M_PI / 2.0) / head_dim;

    // 4 centroids for 2-bit MSE (bits=3 → mse_bits=2 → 4 levels)
    m.k_centroids = 4;
    float sqrt_d = sqrt((float)head_dim);
    m.centroids = { -1.51f/sqrt_d, -0.453f/sqrt_d, 0.453f/sqrt_d, 1.51f/sqrt_d };

    // Random Pi and S matrices
    srand(42);
    m.pi.resize(head_dim * head_dim);
    m.s.resize(head_dim * head_dim);
    for (int i = 0; i < head_dim * head_dim; i++) {
        m.pi[i] = randf() * 0.1f;
        m.s[i] = randf();
    }
    // Make Pi approximately orthogonal (QR-like: just normalize rows)
    for (int i = 0; i < head_dim; i++) {
        float norm = 0;
        for (int j = 0; j < head_dim; j++) norm += m.pi[i*head_dim+j] * m.pi[i*head_dim+j];
        norm = sqrt(norm);
        if (norm > 1e-6) {
            for (int j = 0; j < head_dim; j++) m.pi[i*head_dim+j] /= norm;
        }
    }

    turboquant::row_layout layout = turboquant::compute_row_layout(m);
    printf("Layout: idx_bytes=%u, norm_bytes=%u, sign_bytes=%u, gamma_bytes=%u, total=%u\n",
           layout.idx_bytes, layout.norm_bytes, layout.sign_bytes, layout.gamma_bytes, layout.total_bytes);
    printf("Bits per idx coord: %d, k_centroids: %d\n\n", layout.idx_bits_per_coord, m.k_centroids);

    // Test 1: Single vector roundtrip
    printf("--- Test 1: Single vector roundtrip ---\n");
    std::vector<float> src(head_dim), dst(head_dim);
    for (int i = 0; i < head_dim; i++) src[i] = randf();

    std::vector<uint8_t> packed(layout.total_bytes);
    turboquant::quantize_head_row(m, layout, src.data(), packed.data());
    turboquant::dequant_head_row(m, layout, packed.data(), dst.data());

    float mse = 0, src_energy = 0;
    for (int i = 0; i < head_dim; i++) {
        float diff = src[i] - dst[i];
        mse += diff * diff;
        src_energy += src[i] * src[i];
    }
    mse /= head_dim;
    float snr = 10 * log10(src_energy / (mse * head_dim + 1e-10));
    printf("  MSE: %.6f, SNR: %.1f dB, src_energy: %.4f\n", mse, snr, src_energy);
    printf("  src[0:5]: %.4f %.4f %.4f %.4f %.4f\n", src[0], src[1], src[2], src[3], src[4]);
    printf("  dst[0:5]: %.4f %.4f %.4f %.4f %.4f\n", dst[0], dst[1], dst[2], dst[3], dst[4]);

    // Test 2: Multiple vectors — check average quality
    printf("\n--- Test 2: 1000 random vectors ---\n");
    float total_mse = 0, total_energy = 0;
    int n_test = 1000;
    for (int t = 0; t < n_test; t++) {
        for (int i = 0; i < head_dim; i++) src[i] = randf();
        turboquant::quantize_head_row(m, layout, src.data(), packed.data());
        turboquant::dequant_head_row(m, layout, packed.data(), dst.data());
        for (int i = 0; i < head_dim; i++) {
            float diff = src[i] - dst[i];
            total_mse += diff * diff;
            total_energy += src[i] * src[i];
        }
    }
    total_mse /= (n_test * head_dim);
    float avg_snr = 10 * log10(total_energy / (total_mse * n_test * head_dim + 1e-10));
    printf("  Avg MSE: %.6f, Avg SNR: %.1f dB\n", total_mse, avg_snr);

    // Test 3: Cosine similarity preservation (like RSA)
    printf("\n--- Test 3: Cosine similarity preservation ---\n");
    int n_vecs = 50;
    std::vector<std::vector<float>> orig(n_vecs, std::vector<float>(head_dim));
    std::vector<std::vector<float>> recon(n_vecs, std::vector<float>(head_dim));
    for (int v = 0; v < n_vecs; v++) {
        for (int i = 0; i < head_dim; i++) orig[v][i] = randf();
        turboquant::quantize_head_row(m, layout, orig[v].data(), packed.data());
        turboquant::dequant_head_row(m, layout, packed.data(), recon[v].data());
    }

    // Compare pairwise cosine similarities
    float cos_corr = 0;
    int n_pairs = 0;
    for (int a = 0; a < n_vecs; a++) {
        for (int b = a+1; b < n_vecs; b++) {
            float dot_orig = 0, dot_recon = 0;
            float na = 0, nb = 0, ra = 0, rb = 0;
            for (int i = 0; i < head_dim; i++) {
                dot_orig += orig[a][i] * orig[b][i];
                dot_recon += recon[a][i] * recon[b][i];
                na += orig[a][i] * orig[a][i];
                nb += orig[b][i] * orig[b][i];
                ra += recon[a][i] * recon[a][i];
                rb += recon[b][i] * recon[b][i];
            }
            float cos_orig = dot_orig / (sqrt(na * nb) + 1e-10);
            float cos_recon = dot_recon / (sqrt(ra * rb) + 1e-10);
            cos_corr += (cos_orig - cos_recon) * (cos_orig - cos_recon);
            n_pairs++;
        }
    }
    cos_corr = sqrt(cos_corr / n_pairs);
    printf("  RMS cosine similarity error: %.6f (%d pairs)\n", cos_corr, n_pairs);

    // Test 4: Check that the from_float/to_float roundtrip works (simulating GGML path)
    printf("\n--- Test 4: Batch quantize/dequant (GGML path simulation) ---\n");
    int n_heads = 8;
    int n_elem = n_heads * head_dim;
    std::vector<float> batch_src(n_elem), batch_dst(n_elem);
    for (int i = 0; i < n_elem; i++) batch_src[i] = randf();

    // Quantize: n_heads blocks of head_dim
    std::vector<uint8_t> batch_packed(n_heads * layout.total_bytes);
    for (int h = 0; h < n_heads; h++) {
        turboquant::quantize_head_row(m, layout,
            batch_src.data() + h * head_dim,
            batch_packed.data() + h * layout.total_bytes);
    }

    // Dequant
    for (int h = 0; h < n_heads; h++) {
        turboquant::dequant_head_row(m, layout,
            batch_packed.data() + h * layout.total_bytes,
            batch_dst.data() + h * head_dim);
    }

    float batch_mse = 0;
    for (int i = 0; i < n_elem; i++) {
        float diff = batch_src[i] - batch_dst[i];
        batch_mse += diff * diff;
    }
    batch_mse /= n_elem;
    printf("  Batch MSE: %.6f (8 heads × 128 dim)\n", batch_mse);
    printf("  head0 src[0:3]: %.4f %.4f %.4f\n", batch_src[0], batch_src[1], batch_src[2]);
    printf("  head0 dst[0:3]: %.4f %.4f %.4f\n", batch_dst[0], batch_dst[1], batch_dst[2]);

    // Test 5: Zero vector
    printf("\n--- Test 5: Edge cases ---\n");
    for (int i = 0; i < head_dim; i++) src[i] = 0.0f;
    turboquant::quantize_head_row(m, layout, src.data(), packed.data());
    turboquant::dequant_head_row(m, layout, packed.data(), dst.data());
    float zero_err = 0;
    for (int i = 0; i < head_dim; i++) zero_err += dst[i] * dst[i];
    printf("  Zero input → output energy: %.8f (should be ~0)\n", zero_err);

    // Constant vector
    for (int i = 0; i < head_dim; i++) src[i] = 1.0f;
    turboquant::quantize_head_row(m, layout, src.data(), packed.data());
    turboquant::dequant_head_row(m, layout, packed.data(), dst.data());
    float const_mse_val = 0;
    for (int i = 0; i < head_dim; i++) {
        float diff = src[i] - dst[i];
        const_mse_val += diff * diff;
    }
    printf("  Constant(1.0) roundtrip MSE: %.6f\n", const_mse_val / head_dim);

    printf("\n=== Done ===\n");
    return 0;
}
