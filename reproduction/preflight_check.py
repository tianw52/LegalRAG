#!/usr/bin/env python3
"""Pre-flight check: verifies all dependencies and model caches are ready
before submitting SLURM jobs.

Usage:
    python reproduction/preflight_check.py

Exits 0 if everything is ready, 1 if anything is missing.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

VENV = Path("/home/ram112/projects/def-jieliang/ram112/PyTorch")

E5_MODELS = [
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
    "intfloat/e5-mistral-7b-instruct",
]

HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "✓" if ok else "✗"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {label}{suffix}")
    return ok


def hub_dir(model_id: str) -> Path:
    return HF_CACHE / ("models--" + model_id.replace("/", "--"))


def model_cached(model_id: str) -> bool:
    d = hub_dir(model_id)
    if not d.exists():
        return False
    # Must have a snapshots subdirectory with at least one entry
    snapshots = d / "snapshots"
    if not snapshots.exists():
        return False
    snaps = list(snapshots.iterdir())
    if not snaps:
        return False
    # Each snapshot should contain model files (safetensors or pytorch_model.bin)
    snap = snaps[0]
    return any(
        f.suffix in (".safetensors", ".bin") or f.name.endswith("model.safetensors")
        for f in snap.iterdir()
        if f.is_file()
    )


def main() -> int:
    print()
    print("=" * 60)
    print("  Table 22 Baseline — Pre-flight Check")
    print("=" * 60)
    print()

    all_ok = True

    print("  Python packages:")
    print("  NOTE: 'module load arrow/17.0.0' must be active for pyarrow to be available.")
    print()
    for pkg, name in [
        ("rank_bm25",           "rank_bm25"),
        ("numpy",               "numpy"),
        ("pyarrow",             "pyarrow (via arrow/17.0.0 module)"),
        ("sentence_transformers","sentence_transformers"),
        ("datasets",            "datasets"),
        ("tqdm",                "tqdm"),
    ]:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            ok = check(name, True, ver)
        except ImportError as e:
            ok = check(name, False, str(e))
        all_ok = all_ok and ok

    # faiss is optional — numpy fallback exists
    try:
        import faiss
        check("faiss (optional)", True, faiss.__version__)
    except ImportError:
        check("faiss (optional)", False, "will use numpy fallback — OK")

    print()
    print("  HuggingFace model cache:")
    print(f"  Cache dir: {HF_CACHE}")
    for m in E5_MODELS:
        cached = model_cached(m)
        detail = str(hub_dir(m)) if not cached else "cached"
        ok = check(m, cached, detail)
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("  ✓  All checks passed — safe to submit SLURM jobs.")
        print()
        return 0
    else:
        print("  ✗  Some checks failed — fix issues above before submitting.")
        print()
        print("  To install missing packages:")
        print("    source {}/bin/activate".format(VENV))
        print("    pip install rank-bm25")
        print()
        print("  To download missing E5 models:")
        print("    python3 reproduction/preflight_check.py  # re-run after download")
        print("    python3 /tmp/download_e5.py              # or trigger download directly")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
