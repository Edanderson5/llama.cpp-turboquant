#include "llama-turboquant.h"
#include <cstdio>
#include <cmath>
#include <vector>

extern "C" {
    struct ggml_tensor;
    size_t ggml_type_size(int) { return 0; }
    void ggml_backend_tensor_get(ggml_tensor*, void*, size_t, size_t) {}
    void ggml_backend_tensor_set(ggml_tensor*, const void*, size_t, size_t) {}
    void ggml_set_type_traits_funcs(int, void*, void*) {}
    void ggml_set_type_traits_cpu_from_float(int, void*) {}
}

int main(int argc, char** argv) {
    auto m = turboquant::load_meta(argv[1]);
    int d = m.head_dim;

    printf("centroids[%d]:", m.k_centroids);
    for (int i = 0; i < m.k_centroids; i++) printf(" %.8f", m.centroids[i]);
    printf("\n");

    // Test the rotation step in detail
    // Input: unit vector e_0 = [1, 0, 0, ...]
    std::vector<float> x(d, 0.0f);
    x[0] = 1.0f;

    // y = x @ Pi^T: y[j] = sum_i x[i] * Pi[i*d + j]
    // For e_0: y[j] = Pi[0*d + j] = Pi[0][j]
    printf("\nRotation test (e_0):\n");
    printf("Pi[0][0:5]: %.6f %.6f %.6f %.6f %.6f\n",
           m.pi[0], m.pi[1], m.pi[2], m.pi[3], m.pi[4]);

    std::vector<float> y(d);
    for (int j = 0; j < d; j++) {
        float s = 0;
        for (int i = 0; i < d; i++) s += x[i] * m.pi[i*d + j];
        y[j] = s;
    }
    printf("y = x @ Pi^T: y[0:5] = %.6f %.6f %.6f %.6f %.6f\n",
           y[0], y[1], y[2], y[3], y[4]);

    // Check y norm (should be 1 if Pi is orthogonal)
    float yn = 0;
    for (int i = 0; i < d; i++) yn += y[i]*y[i];
    printf("||y||^2 = %.6f (should be 1.0)\n", yn);

    // Quantize y: find nearest centroid per coordinate
    std::vector<int> idx(d);
    std::vector<float> y_tilde(d);
    for (int j = 0; j < d; j++) {
        int best = 0;
        float best_d = fabs(y[j] - m.centroids[0]);
        for (int c = 1; c < m.k_centroids; c++) {
            float dist = fabs(y[j] - m.centroids[c]);
            if (dist < best_d) { best_d = dist; best = c; }
        }
        idx[j] = best;
        y_tilde[j] = m.centroids[best];
    }

    // MSE in rotated space
    float mse_rot = 0;
    for (int j = 0; j < d; j++) {
        float diff = y[j] - y_tilde[j];
        mse_rot += diff * diff;
    }
    printf("\nMSE in rotated space: %.6f (should be small)\n", mse_rot / d);
    printf("y[0:5]:       %.6f %.6f %.6f %.6f %.6f\n", y[0], y[1], y[2], y[3], y[4]);
    printf("y_tilde[0:5]: %.6f %.6f %.6f %.6f %.6f\n", y_tilde[0], y_tilde[1], y_tilde[2], y_tilde[3], y_tilde[4]);

    // Inverse rotate: x_tilde = y_tilde @ Pi
    // x_tilde[i] = sum_j y_tilde[j] * Pi[j*d + i]
    std::vector<float> x_tilde(d);
    for (int i = 0; i < d; i++) {
        float s = 0;
        for (int j = 0; j < d; j++) s += y_tilde[j] * m.pi[j*d + i];
        x_tilde[i] = s;
    }
    printf("\nx_tilde (MSE only, no QJL):\n");
    printf("x_tilde[0:5]: %.6f %.6f %.6f %.6f %.6f\n",
           x_tilde[0], x_tilde[1], x_tilde[2], x_tilde[3], x_tilde[4]);
    float mse_nojql = 0;
    for (int i = 0; i < d; i++) {
        float diff = x[i] - x_tilde[i];
        mse_nojql += diff * diff;
    }
    printf("MSE (no QJL): %.6f\n", mse_nojql / d);

    // Full roundtrip via our functions
    auto layout = turboquant::compute_row_layout(m);
    std::vector<uint8_t> packed(layout.total_bytes);
    std::vector<float> dst(d);
    turboquant::quantize_head_row(m, layout, x.data(), packed.data());
    turboquant::dequant_head_row(m, layout, packed.data(), dst.data());
    printf("\nFull roundtrip:\n");
    printf("dst[0:5]: %.6f %.6f %.6f %.6f %.6f\n", dst[0], dst[1], dst[2], dst[3], dst[4]);
    float full_mse = 0;
    for (int i = 0; i < d; i++) { float diff = x[i] - dst[i]; full_mse += diff*diff; }
    printf("Full MSE: %.6f\n", full_mse / d);

    return 0;
}
