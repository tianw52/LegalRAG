#!/usr/bin/env bash
# Serve Qwen3 with vLLM on the lab machine (secb1010-d11, RTX 4090 24 GB) as an
# OpenAI-compatible API for the exploration app.
#
# One-time setup on the lab machine (separate venv — vLLM pins its own torch). Use the
# local NVMe, not the NFS home (setup used 2026-10-18; vLLM 0.32.0):
#   mkdir -p /local-scratch/$USER && cd /local-scratch/$USER
#   python3 -m venv vllm-env && vllm-env/bin/pip install uv && vllm-env/bin/uv pip install --python vllm-env/bin/python vllm
#
# Run on the lab machine (activate the venv: warm-up needs its `ninja` on PATH):
#   source /local-scratch/$USER/vllm-env/bin/activate
#   export HF_HOME=/local-scratch/$USER/hf                 # model cache on local disk
#   bash scripts/serve_qwen3_vllm.sh                       # Qwen3-30B-A3B GPTQ-Int4 (~16 GB, KV ≈ 46k tokens)
#   bash scripts/serve_qwen3_vllm.sh Qwen/Qwen3-14B-AWQ    # lighter fallback (~10 GB)
#
# From the laptop, tunnel the port (keep this running):
#   ssh -N -o ServerAliveInterval=15 -L 8001:127.0.0.1:8001 secb1010-d11.csil.sfu.ca
#
# The laptop's .env vLLM block (the default in .env.example) then uses it:
#   LLM_BASE_URL=http://localhost:8001/v1
#   LLM_API_KEY=EMPTY
#   LLM_MODEL=qwen3            # = --served-model-name below
#   LLM_THINKING_OFF=vllm      # app switches Qwen3 thinking off per request
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
