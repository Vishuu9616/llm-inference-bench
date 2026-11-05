"""Plot one or more result files: throughput and p95 TTFT vs concurrency."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="result JSON files to compare")
    p.add_argument("--out", default="results/comparison.png")
    args = p.parse_args()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for f in args.files:
        d = json.loads(Path(f).read_text())
        xs = [r["concurrency"] for r in d["results"]]
        a1.plot(xs, [r["output_tok_per_s"] for r in d["results"]], marker="o", label=d["label"])
        a2.plot(xs, [r["ttft_p95_s"] for r in d["results"]], marker="o", label=d["label"])
    a1.set(xlabel="Concurrency", ylabel="Output tokens/s", title="Throughput")
    a2.set(xlabel="Concurrency", ylabel="TTFT p95 (s)", title="Time to first token")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
