"""Deterministic synthetic prompts so runs are comparable across configs."""
import random

_WORDS = ("satellite model latency throughput batch kernel cache token memory "
          "gradient tensor pipeline inference cluster request scheduler decode "
          "prefill quantization attention layer weight sampling stream queue").split()


def build_prompt(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    filler = " ".join(rng.choice(_WORDS) for _ in range(n_words))
    return "Summarize the following text in three sentences.\n\n" + filler
