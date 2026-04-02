// Test TurboQuant roundtrip using actual .tqmeta sidecar
#include "llama-turboquant.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Stubs for ggml functions we don't need
extern "C" {
    struct ggml_tensor;
    size_t ggml_type_size(int) { return 0; }
    void ggml_backend_tensor_get(ggml_tensor*, void*, size_t, size_t) {}
    void ggml_backend_tensor_set(ggml_tensor*, const void*, size_t, size_t) {}
    void ggml_set_type_traits_funcs(int, void*, void*) {}
    void ggml_set_type_traits_cpu_from_float(int, void*) {}
}

static float randf() { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("Usage: %s <path.tqmeta>\n", argv[0]);
        return 1;
    }

    printf("=== TurboQuant Sidecar Roundtrip Test ===\n\n");

    turboquant::meta m = turboquant::load_meta(argv[1]);
    printf("Loaded: bits=%.1f head_dim=%d k_centroids=%d qjl_factor=%.8f\n",
           m.bits, m.head_dim, m.k_centroids, m.qjl_factor);

    // Verify Pi is orthogonal: Pi @ Pi^T should be ~identity
    int d = m.head_dim;
    float max_offdiag = 0, min_diag = 1e9;
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            float dot = 0;
            for (int k = 0; k < d; k++) {
                dot += m.pi[i*d+k] * m.pi[j*d+k];
            }
            if (i == j) { if (dot < min_diag) min_diag = dot; }
            else { if (fabs(dot) > max_offdiag) max_offdiag = fabs(dot); }
        }
    }
    printf("Pi orthogonality: min_diag=%.6f max_offdiag=%.6f\n", min_diag, max_offdiag);

    turboquant::row_layout layout = turboquant::compute_row_layout(m);
    printf("Layout: total=%u bytes/head (%.1f bits/element)\n\n",
           layout.total_bytes, 8.0f * layout.total_bytes / d);

    srand(42);
    std::vector<float> src(d), dst(d);
    std::vector<uint8_t> packed(layout.total_bytes);

    // Test 1: Average roundtrip quality
    printf("--- Test 1: 1000 random vectors ---\n");
    float total_mse = 0, total_energy = 0;
    int n_test = 1000;
    for (int t = 0; t < n_test; t++) {
        for (int i = 0; i < d; i++) src[i] = randf();
        turboquant::quantize_head_row(m, layout, src.data(), packed.data());
        turboquant::dequant_head_row(m, layout, packed.data(), dst.data());
        for (int i = 0; i < d; i++) {
            float diff = src[i] - dst[i];
            total_mse += diff * diff;
            total_energy += src[i] * src[i];
        }
    }
    total_mse /= (n_test * d);
    float avg_snr = 10 * log10(total_energy / (total_mse * n_test * d + 1e-10));
    printf("  Avg MSE: %.6f, Avg SNR: %.1f dB\n", total_mse, avg_snr);

    // Show first vector
    for (int i = 0; i < d; i++) src[i] = randf();
    turboquant::quantize_head_row(m, layout, src.data(), packed.data());
    turboquant::dequant_head_row(m, layout, packed.data(), dst.data());
    printf("  src[0:5]: %.4f %.4f %.4f %.4f %.4f\n", src[0], src[1], src[2], src[3], src[4]);
    printf("  dst[0:5]: %.4f %.4f %.4f %.4f %.4f\n", dst[0], dst[1], dst[2], dst[3], dst[4]);

    // Test 2: Cosine similarity
    printf("\n--- Test 2: Cosine similarity preservation (50 vectors) ---\n");
    int n_vecs = 50;
    std::vector<std::vector<float>> orig(n_vecs, std::vector<float>(d));
    std::vector<std::vector<float>> recon(n_vecs, std::vector<float>(d));
    for (int v = 0; v < n_vecs; v++) {
        for (int i = 0; i < d; i++) orig[v][i] = randf();
        turboquant::quantize_head_row(m, layout, orig[v].data(), packed.data());
        turboquant::dequant_head_row(m, layout, packed.data(), recon[v].data());
    }

    float cos_errs = 0; int np = 0;
    for (int a = 0; a < n_vecs; a++) {
        for (int b = a+1; b < n_vecs; b++) {
            float dot_o = 0, dot_r = 0, na = 0, nb = 0, ra = 0, rb = 0;
            for (int i = 0; i < d; i++) {
                dot_o += orig[a][i] * orig[b][i];
                dot_r += recon[a][i] * recon[b][i];
                na += orig[a][i]*orig[a][i]; nb += orig[b][i]*orig[b][i];
                ra += recon[a][i]*recon[a][i]; rb += recon[b][i]*recon[b][i];
            }
            float co = dot_o / (sqrt(na*nb)+1e-10);
            float cr = dot_r / (sqrt(ra*rb)+1e-10);
            cos_errs += (co-cr)*(co-cr);
            np++;
        }
    }
    printf("  RMS cosine error: %.6f\n", sqrt(cos_errs/np));

    // Test 3: Realistic scale (model activations are typically ~0.01-0.1)
    printf("\n--- Test 3: Realistic activation scale ---\n");
    total_mse = 0; total_energy = 0;
    for (int t = 0; t < n_test; t++) {
        for (int i = 0; i < d; i++) src[i] = randf() * 0.05f;  // typical activation scale
        turboquant::quantize_head_row(m, layout, src.data(), packed.data());
        turboquant::dequant_head_row(m, layout, packed.data(), dst.data());
        for (int i = 0; i < d; i++) {
            float diff = src[i] - dst[i];
            total_mse += diff * diff;
            total_energy += src[i] * src[i];
        }
    }
    total_mse /= (n_test * d);
    avg_snr = 10 * log10(total_energy / (total_mse * n_test * d + 1e-10));
    printf("  Small-scale MSE: %.8f, SNR: %.1f dB\n", total_mse, avg_snr);

    printf("\n=== Done ===\n");
    return 0;
}
