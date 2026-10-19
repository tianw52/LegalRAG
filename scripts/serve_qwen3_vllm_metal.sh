#!/usr/bin/env bash
# Serve Qwen3-8B with vLLM on an Apple Silicon Mac (vllm-metal plugin, MLX on the GPU)
# as the app's OpenAI-compatible LLM — same port/name as the lab vLLM, so the .env
# vLLM block works unchanged (LLM_BASE_URL=http://localhost:8001/v1, LLM_MODEL=qwen3,
# LLM_THINKING_OFF=vllm).  Unlike Ollama, it runs the app's parallel calls concurrently.
#
# One-time install (Homebrew; private Python env, no activation needed):
#   brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
#   brew install vllm-project/vllm-metal/vllm-metal
#
# Memory (24 GB M2): weights ~4.6 GB + KV pool.  At 0.45 the pool (2.2 GB) can't hold one
# 16k-token request; 0.6 gives ~33k tokens (≈6 concurrent ~5k-token view calls).
# Stop Ollama first (`ollama stop qwen3:8b`) — both together swap heavily.
set -euo pipefail

MODEL="${1:-mlx-community/Qwen3-8B-4bit}"
PORT="${PORT:-8001}"
MEM="${MEM:-0.6}"

exec vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --served-model-name qwen3 \
    --max-model-len 16384 \
    --gpu-memory-utilization "$MEM"
