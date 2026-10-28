import sys

import numpy as np


def load(name, variant):
    return np.load(f"runs/sweep_{name}/per_query_{variant}.npy")


def compare(a, b, variant="plain", n=5000):
    x, y = load(a, variant), load(b, variant)
    k = min(x.shape[1], y.shape[1])
    rng = np.random.default_rng(0)
    for row, metric in enumerate(("nDCG@10", "MRR")):
        d = x[row, :k] - y[row, :k]
        boot = [d[rng.integers(0, k, k)].mean() for _ in range(n)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{metric}: {a} - {b} = {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  over {k} queries")


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "plain")
