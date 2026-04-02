#!/bin/bash
# Test TurboQuant quality: compare baseline vs TQ outputs
cd ~/llama.cpp-turboquant

GGUF=qwen3-8b.gguf
META=qwen2.5-3b.tqmeta
PORT_BASE=8090
PORT_TQ=8091

echo "=== Starting baseline server ==="
./build/bin/llama-server -m $GGUF -c 2048 --port $PORT_BASE --host 0.0.0.0 -ngl 0 &
PID_BASE=$!
sleep 10

echo "=== Starting TurboQuant server ==="
./build/bin/llama-server -m $GGUF -c 2048 --port $PORT_TQ --host 0.0.0.0 -ngl 0 --turboquant-meta $META &
PID_TQ=$!
sleep 10

echo "=== Testing prompts ==="
for prompt in \
  "What is photosynthesis?" \
  "Explain gravity in one sentence." \
  "What causes rain?" \
  "How does a computer store data?" \
  "What is the speed of light?"; do

  echo ""
  echo "--- PROMPT: $prompt ---"

  echo "BASELINE:"
  curl -s http://localhost:$PORT_BASE/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":80,\"temperature\":0}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:200])" 2>/dev/null
  echo ""

  echo "TURBOQUANT:"
  curl -s http://localhost:$PORT_TQ/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":80,\"temperature\":0}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:200])" 2>/dev/null
  echo ""
done

echo "=== Cleanup ==="
kill $PID_BASE $PID_TQ 2>/dev/null
wait 2>/dev/null
echo "Done."
