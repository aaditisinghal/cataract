"""Aggregate per-seed result JSONs into paper-ready mean ± 95% CI + Holm-corrected significance (plan A2).

Every headline experiment is run across N seeds (the seed sweep). This script gathers those N result
JSONs for ONE experiment, pulls the headline metric(s) out of each by dotted key path, and reports for
each metric the across-seed mean with a 95% percentile-bootstrap CI. It then computes a two-sided
bootstrap p-value that each metric differs from a null (default 0 — i.e. "is this delta real?") and
Holm-Bonferroni corrects those p-values across the whole metric set so the paper's primary claim family
(name / id / dob recovery deltas, defense-dominance deltas) is protected against multiple-comparison
false positives.

Sources for ``--runs``: a local directory (recursively globbed for ``*.json``), an explicit glob
pattern, or a ``gs://`` prefix (downloaded via ``_gcs_download`` then globbed). Point it at the subdir
that holds only the per-seed JSONs of a single experiment so the metric keys resolve uniformly.

Example (after a seed sweep of the retrieval attack has landed)::

  python3 -m experiments.aggregate_seeds \
      --runs gs://patchguard-reakon-artifacts/runs \
      --metric-keys summary.name.top1_acc,summary.id_no.top1_acc,summary.dob.top1_acc
"""

from __future__ import annotations

import argparse
import glob
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _dig(obj: Any, dotted: str) -> Any:
    """Navigate ``obj`` by a dotted path; list segments are integer-indexed. Raises KeyError/IndexError."""
    cur = obj
    for tok in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(tok)]
        elif isinstance(cur, dict):
            if tok not in cur:
                raise KeyError(dotted)
            cur = cur[tok]
        else:
            raise KeyError(dotted)
    return cur


def _resolve_run_files(runs: str, tmp: Path) -> list[Path]:
    """Resolve ``--runs`` (gs:// prefix | glob | dir | single file) to a sorted list of JSON paths."""
    s = str(runs)
    if s.startswith("gs://"):
        from experiments.train_funsd import _gcs_download

        local = _gcs_download(s, tmp)
        return sorted(local.rglob("*.json"))
    if any(ch in s for ch in "*?[]"):
        return sorted(Path(p) for p in glob.glob(s, recursive=True))
    p = Path(s)
    if p.is_dir():
        return sorted(p.rglob("*.json"))
    return [p] if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="dir | glob | gs:// prefix holding per-seed JSONs")
    ap.add_argument("--metric-keys", required=True, help="comma-separated dotted paths into each JSON")
    ap.add_argument("--out", default="results/aggregate_seeds")
    ap.add_argument("--null", type=float, default=0.0, help="H0 value each metric is tested against")
    ap.add_argument("--iters", type=int, default=10_000, help="bootstrap resamples")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from patchguard.eval.stats import bootstrap_ci, bootstrap_pvalue, holm_bonferroni
    from patchguard.repro import run_fingerprint, seed_everything
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)
    tmp_dl = Path(tempfile.mkdtemp())

    keys = [k.strip() for k in args.metric_keys.split(",") if k.strip()]
    files = _resolve_run_files(args.runs, tmp_dl)

    per_key: dict[str, list[float]] = {k: [] for k in keys}
    used_files: list[str] = []
    for f in files:
        try:
            payload = json.loads(Path(f).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        got_any = False
        for k in keys:
            try:
                per_key[k].append(float(_dig(payload, k)))
                got_any = True
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        if got_any:
            used_files.append(str(f))

    metrics: dict[str, dict[str, Any]] = {}
    pvals: dict[str, float] = {}
    for k in keys:
        vals = np.asarray(per_key[k], dtype=float)
        if vals.size == 0:
            metrics[k] = {"n": 0, "note": "metric not found in any run"}
            continue
        mean, lo, hi = bootstrap_ci(vals, iters=args.iters, alpha=args.alpha, seed=args.seed)
        entry: dict[str, Any] = {
            "n": int(vals.size),
            "per_seed": vals.tolist(),
            "mean": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "null": args.null,
        }
        if vals.size > 1:
            p = bootstrap_pvalue(vals, null=args.null, iters=args.iters, seed=args.seed)
            entry["boot_p"] = p
            pvals[k] = p
        metrics[k] = entry

    holm = holm_bonferroni(pvals, alpha=args.alpha) if pvals else {}
    for k, res in holm.items():
        metrics[k]["p_adj"] = res["p_adj"]
        metrics[k]["reject"] = res["reject"]

    n_seeds = len(used_files)
    print(f"=== AGGREGATE ({n_seeds} seed files) ===")
    for k in keys:
        m = metrics[k]
        if m.get("n", 0) == 0:
            print(f"  {k}: MISSING")
            continue
        line = f"  {k}: mean={m['mean']:.4f}  95% CI [{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]  n={m['n']}"
        if "p_adj" in m:
            line += f"  p_adj={m['p_adj']:.4g} {'REJECT-H0' if m['reject'] else 'ns'}"
        print(line)

    payload = {
        "mode": "aggregate_seeds",
        "runs": str(args.runs),
        "metric_keys": keys,
        "n_seeds": n_seeds,
        "seed_files": used_files,
        "null_value": args.null,
        "alpha": args.alpha,
        "metrics": metrics,
        "holm": holm,
        "fingerprint": run_fingerprint(),
    }
    (local_out / "aggregate_seeds.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote aggregate_seeds.json -> {args.out}")


if __name__ == "__main__":
    main()
