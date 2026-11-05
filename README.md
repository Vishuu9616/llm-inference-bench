# llm-inference-bench

A small, dependency-light toolkit for **benchmarking LLM serving** behind any
OpenAI-compatible API: [vLLM](https://github.com/vllm-project/vllm),
[SGLang](https://github.com/sgl-project/sglang), or a hosted endpoint such as
Nebius Token Factory.

It sweeps concurrency levels and reports the numbers that matter when tuning
inference: **time-to-first-token (TTFT)**, **end-to-end latency (p50/p95)**,
**decode speed per request** and **aggregate output tokens/s**. Use it to compare
configurations (fp16 vs. AWQ int4, different batch limits, different engines or
models) on equal footing.

## Why this exists

I train and evaluate vision models on small GPU budgets (Kaggle T4s), so I care
about getting more out of limited hardware. This repo is my workflow for
measuring serving trade-offs instead of guessing them.

## Quick start

```bash
git clone https://github.com/<your-username>/llm-inference-bench
cd llm-inference-bench
pip install -r requirements.txt

# 1) start a server (needs a GPU and `pip install vllm`)
bash scripts/serve_vllm.sh Qwen/Qwen2.5-1.5B-Instruct half

# 2) benchmark it
python -m bench.benchmark --model Qwen/Qwen2.5-1.5B-Instruct \
    --label fp16 --output results/fp16.json

# 3) plot / compare runs
python -m bench.plot results/fp16.json results/awq.json
```

No GPU handy? Test the tool itself against the built-in mock server:

```bash
uvicorn tests.mock_server:app --port 8000 &
python -m bench.benchmark --model mock --label mock --num-requests 16 --concurrency 1,4
```

For a hosted endpoint, point `--base-url` at it and put the key in an environment
variable (`--api-key-env`). Never commit keys.

## What gets measured

| Metric | Meaning |
|---|---|
| `ttft_p50_s` / `ttft_p95_s` | Time until the first streamed token arrives |
| `latency_p50_s` / `latency_p95_s` | Full request time |
| `decode_tok_per_s_per_req_p50` | Tokens/s after the first token, per request |
| `output_tok_per_s` | Total output tokens / wall time (system throughput) |
| `req_per_s`, `failed` | Request rate and failure count |

Token counts use the server's `usage` field when available and fall back to the
streamed chunk count. Prompts are synthetic and seeded, so runs are comparable.

## Experiments

Fill this table with your own measured results from `results/*.json`
(model, GPU, vLLM version, settings), then add the plot and your takeaways.

| Config | Concurrency | Output tok/s | TTFT p95 (s) | Latency p95 (s) |
|---|---|---|---|---|
| _your run_ | | | | |

![comparison](results/comparison.png)

## Project layout

```
bench/benchmark.py   # async concurrency sweep + metrics
bench/prompts.py     # deterministic synthetic prompts
bench/plot.py        # throughput / TTFT comparison plots
scripts/serve_vllm.sh
tests/mock_server.py # offline OpenAI-compatible streaming mock (FastAPI)
```

## Ideas / roadmap

- [ ] Compare fp16 vs. AWQ/GPTQ int4 on the same model
- [ ] Sweep vLLM `--max-num-seqs` and prefix caching
- [ ] Add SGLang as a second engine
- [ ] Quality check (e.g. a small eval set) alongside speed

## License

MIT
