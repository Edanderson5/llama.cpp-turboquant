#!/bin/bash
# Patch Ollama's llama.cpp with TurboQuant support
# Usage: ./ollama-patch.sh /path/to/ollama-source
set -e

OL="${1:?Usage: $0 /path/to/ollama-source}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Patching Ollama at $OL with TurboQuant ==="

# 1. Copy new TurboQuant files
echo "Copying new files..."
cp "$SCRIPT_DIR/src/llama-turboquant.h"  "$OL/llama/llama.cpp/src/"
cp "$SCRIPT_DIR/src/llama-turboquant.cpp" "$OL/llama/llama.cpp/src/"
cp "$SCRIPT_DIR/ggml/src/ggml-cuda/turboquant.cu"  "$OL/ml/backend/ggml/ggml/src/ggml-cuda/"
cp "$SCRIPT_DIR/ggml/src/ggml-cuda/turboquant.cuh" "$OL/ml/backend/ggml/ggml/src/ggml-cuda/"
cp "$SCRIPT_DIR/ggml/src/ggml-cuda/hadamard.cuh"   "$OL/ml/backend/ggml/ggml/src/ggml-cuda/"
cp "$SCRIPT_DIR/ggml/src/ggml-cuda/fattn-tq3.cu"   "$OL/ml/backend/ggml/ggml/src/ggml-cuda/"

# 2. Patch ggml.h — add TQ3_0 type
GGML_H="$OL/ml/backend/ggml/ggml/include/ggml.h"
if ! grep -q "GGML_TYPE_TQ3_0" "$GGML_H"; then
    # Add TQ3_0 before GGML_TYPE_COUNT
    sed -i '/GGML_TYPE_COUNT/i\        GGML_TYPE_TQ3_0   = '"$(grep -oP 'GGML_TYPE_COUNT\s*=\s*\K\d+' "$GGML_H")"', // TurboQuant 3-bit KV cache' "$GGML_H"
    # Increment GGML_TYPE_COUNT
    OLD_COUNT=$(grep -oP 'GGML_TYPE_COUNT\s*=\s*\K\d+' "$GGML_H")
    NEW_COUNT=$((OLD_COUNT + 1))
    sed -i "s/GGML_TYPE_COUNT   = $OLD_COUNT/GGML_TYPE_COUNT   = $NEW_COUNT/" "$GGML_H"
    echo "  Added GGML_TYPE_TQ3_0 = $OLD_COUNT (count now $NEW_COUNT)"
fi

# Add setter function declarations
if ! grep -q "ggml_set_type_traits_funcs" "$GGML_H"; then
    sed -i '/ggml_get_type_traits/a\    GGML_API void ggml_set_type_traits_funcs(enum ggml_type type, ggml_to_float_t to_float, ggml_from_float_t from_float);\n    GGML_API void ggml_set_type_traits_size(enum ggml_type type, int64_t blck_size, size_t type_size);' "$GGML_H"
    echo "  Added ggml_set_type_traits_funcs/size declarations"
fi

# 3. Patch ggml.c — add type traits entry and setter functions
GGML_C="$OL/ml/backend/ggml/ggml/src/ggml.c"
if ! grep -q "GGML_TYPE_TQ3_0" "$GGML_C"; then
    # Make type_traits non-const
    sed -i 's/^static const struct ggml_type_traits type_traits/static struct ggml_type_traits type_traits/' "$GGML_C"
    echo "  Made type_traits non-const"
fi

if ! grep -q "ggml_set_type_traits_funcs" "$GGML_C"; then
    # Add setter after getter
    sed -i '/^const struct ggml_type_traits \* ggml_get_type_traits/,/^}/a\
\
void ggml_set_type_traits_funcs(enum ggml_type type, ggml_to_float_t to_float, ggml_from_float_t from_float) {\
    type_traits[type].to_float = to_float;\
    type_traits[type].from_float_ref = from_float;\
}\
\
void ggml_set_type_traits_size(enum ggml_type type, int64_t blck_size, size_t type_size) {\
    type_traits[type].blck_size = blck_size;\
    type_traits[type].type_size = type_size;\
}' "$GGML_C"
    echo "  Added setter functions to ggml.c"
fi

# 4. Patch ggml-cpu.h — add CPU setter
GGML_CPU_H="$OL/ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.h"
if [ ! -f "$GGML_CPU_H" ]; then
    GGML_CPU_H="$OL/ml/backend/ggml/ggml/include/ggml-cpu.h"
fi
if ! grep -q "ggml_set_type_traits_cpu_from_float" "$GGML_CPU_H"; then
    sed -i '/ggml_get_type_traits_cpu/a\    GGML_BACKEND_API void ggml_set_type_traits_cpu_from_float(enum ggml_type type, ggml_from_float_t from_float);\n    GGML_BACKEND_API void ggml_set_type_traits_cpu_vec_dot(enum ggml_type type, ggml_vec_dot_t vec_dot, enum ggml_type vec_dot_type);' "$GGML_CPU_H"
    echo "  Added CPU setter declarations"
fi

# 5. Patch ggml-cpu.c — add CPU setter implementations
GGML_CPU_C="$OL/ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.c"
if ! grep -q "ggml_set_type_traits_cpu_from_float" "$GGML_CPU_C"; then
    # Make CPU type_traits non-const
    sed -i 's/^static const struct ggml_type_traits_cpu type_traits_cpu/static struct ggml_type_traits_cpu type_traits_cpu/' "$GGML_CPU_C"
    # Add setters after getter
    sed -i '/^const struct ggml_type_traits_cpu \* ggml_get_type_traits_cpu/,/^}/a\
\
void ggml_set_type_traits_cpu_from_float(enum ggml_type type, ggml_from_float_t from_float) {\
    type_traits_cpu[type].from_float = from_float;\
}\
\
void ggml_set_type_traits_cpu_vec_dot(enum ggml_type type, ggml_vec_dot_t vec_dot, enum ggml_type vec_dot_type) {\
    type_traits_cpu[type].vec_dot = vec_dot;\
    type_traits_cpu[type].vec_dot_type = vec_dot_type;\
}' "$GGML_CPU_C"
    echo "  Added CPU setter implementations"
fi

# 6. Patch ops.cpp — add TQ3_0 to switch statements
OPS="$OL/ml/backend/ggml/ggml/src/ggml-cpu/ops.cpp"
if ! grep -q "GGML_TYPE_TQ3_0" "$OPS"; then
    sed -i 's/case GGML_TYPE_TQ2_0:/case GGML_TYPE_TQ2_0:\n        case GGML_TYPE_TQ3_0:/g' "$OPS"
    TQ3_COUNT=$(grep -c "GGML_TYPE_TQ3_0" "$OPS")
    echo "  Added TQ3_0 to $TQ3_COUNT switch cases in ops.cpp"
fi

# 7. Patch llama.h — add turboquant_meta_path
LLAMA_H="$OL/llama/llama.cpp/include/llama.h"
if ! grep -q "turboquant_meta_path" "$LLAMA_H"; then
    sed -i '/enum ggml_type type_v;/a\        const char * turboquant_meta_path; // TurboQuant .tqmeta sidecar path (NULL = disabled)' "$LLAMA_H"
    echo "  Added turboquant_meta_path to llama.h"
fi

# 8. Patch llama-kv-cache.h — add TQ state
KV_H="$OL/llama/llama.cpp/src/llama-kv-cache.h"
if ! grep -q "turboquant" "$KV_H"; then
    sed -i '1s/^/#include "llama-turboquant.h"\n/' "$KV_H"
    echo "  Added turboquant include to llama-kv-cache.h"
    echo "  NOTE: Manual edits needed for init_turboquant() and post_process methods"
fi

echo ""
echo "=== Patch complete ==="
echo "Manual steps remaining:"
echo "  1. Add llama-turboquant.cpp to CMakeLists.txt"
echo "  2. Wire init_turboquant() in llama-context.cpp"
echo "  3. Wire turboquant_post_process() in decode loop"
echo "  4. Add K/V cast for TQ3_0 in llama-graph.cpp"
echo "  5. Build with: go generate ./... && go build ."
