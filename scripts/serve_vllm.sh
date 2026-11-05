#!/usr/bin/env bash
# Serve a model with vLLM's OpenAI-compatible server.
# Example (T4 GPUs have no bf16, so use half precision):
#   bash scripts/serve_vllm.sh Qwen/Qwen2.5-1.5B-Instruct half
#   bash scripts/serve_vllm.sh Qwen/Qwen2.5-1.5B-Instruct-AWQ half awq
set -euo pipefail
MODEL="${1:?model name required}"
DTYPE="${2:-half}"
QUANT="${3:-}"
EXTRA=()
if [[ -n "$QUANT" ]]; then EXTRA+=(--quantization "$QUANT"); fi
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --dtype "$DTYPE" \
  --max-model-len 2048 --gpu-memory-utilization 0.90 \
  --port 8000 "${EXTRA[@]}"
