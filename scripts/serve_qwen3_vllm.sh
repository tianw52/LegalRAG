#!/usr/bin/env bash
# Serve Qwen3 with vLLM on the lab machine (secb1010-d11, RTX 4090 24 GB) as an
# OpenAI-compatible API for the exploration app.
#
# One-time setup on the lab machine (separate venv — vLLM pins its own torch):
#   python3 -m venv ~/vllm-env && source ~/vllm-env/bin/activate && pip install vllm
#
# Run on the lab machine:
#   source ~/vllm-env/bin/activate
#   bash scripts/serve_qwen3_vllm.sh                       # Qwen3-30B-A3B GPTQ-Int4 (~17 GB)
#   bash scripts/serve_qwen3_vllm.sh Qwen/Qwen3-14B-AWQ    # lighter fallback (~10 GB)
#
# From the laptop, tunnel the port (keep this running):
#   ssh -N -L 8001:127.0.0.1:8001 secb1010-d11.csil.sfu.ca
#
# Then in the laptop's .env:
#   LLM_BASE_URL=http://localhost:8001/v1
#   LLM_API_KEY=EMPTY
#   LLM_MODEL=qwen3            # = --served-model-name below
#
# Thinking is disabled per request by the app (explore_policy.yaml → llm.qwen3_no_think).
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}"
PORT="${PORT:-8001}"

exec vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --served-model-name qwen3 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
