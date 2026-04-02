#!/bin/bash
# Benchmark KV cache memory usage: F16 vs TQ3_0
cd ~/llama.cpp-turboquant
GGUF=qwen3-8b.gguf
META=qwen2.5-3b.tqmeta

echo "=== KV Cache Memory Benchmark ==="
echo "Model: Qwen3 8B (Q4_K)"
echo ""

for CTX in 512 2048 8192; do
    echo "--- Context: $CTX ---"

    # F16 baseline
    echo -n "  F16 KV: "
    echo 'Hi' | timeout 30 ./build/bin/llama-cli -m $GGUF -n 1 -c $CTX --log-disable 2>&1 | grep "memory_breakdown\|kv_self\|KV buffer\|Host" | grep "Host" | head -1

    # TQ3_0
    echo -n "  TQ3 KV: "
    echo 'Hi' | timeout 30 ./build/bin/llama-cli -m $GGUF -n 1 -c $CTX --cache-type-k tq3_0 --cache-type-v tq3_0 --turboquant-meta $META --log-disable 2>&1 | grep "memory_breakdown\|kv_self\|KV buffer\|Host" | grep "Host" | head -1

    # Q8_0 for comparison
    echo -n "  Q8 KV:  "
    echo 'Hi' | timeout 30 ./build/bin/llama-cli -m $GGUF -n 1 -c $CTX --cache-type-k q8_0 --cache-type-v q8_0 --log-disable 2>&1 | grep "memory_breakdown\|kv_self\|KV buffer\|Host" | grep "Host" | head -1

    # Q4_0 for comparison
    echo -n "  Q4 KV:  "
    echo 'Hi' | timeout 30 ./build/bin/llama-cli -m $GGUF -n 1 -c $CTX --cache-type-k q4_0 --cache-type-v q4_0 --log-disable 2>&1 | grep "memory_breakdown\|kv_self\|KV buffer\|Host" | grep "Host" | head -1

    echo ""
done
