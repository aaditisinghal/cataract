"""Deploy-cost benchmark for RedactionProjection (MASTER_PLAN / COMPLETION_PLAN C1).

The defense claim is only credible if the transform P is *cheap enough to deploy*. This experiment
quantifies the four costs an operator pays to run ``P`` as an index-time redaction layer, and shows the
three that must be ~0 actually are:

  1. INDEX-TIME latency  — the per-patch cost of applying P once, at index build. This is the ONLY real
     cost. Reported as µs/patch and ms/page (page = ``--n-patches`` patches, ~1030 for ColPali).
  2. PARAMETER storage   — P's own weights (a one-time model artifact, NOT per-page). Reported as a
     param count + bytes, and it amortizes to 0 bytes/page as the corpus grows.
  3. PER-PAGE storage delta — P maps a d-vector to a d-vector, so the stored index is byte-for-byte the
     same size. Added storage per page is therefore 0 (P is *applied*, not appended).
  4. QUERY-TIME overhead — queries use the vanilla frozen encoder; P never touches a query. So the
     retrieval hot path is unchanged: 0 added query latency, 0 added query bytes.

A positive result here is boring by design: "the defense is free at query time, adds no storage, and
costs one small MLP forward per patch at index time." That is exactly the deploy story the paper needs.
This runs on CPU with torch only (no ColPali), so the numbers are produced locally and reported honestly
for whatever hardware the benchmark ran on (see the fingerprint's ``platform``).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path


def benchmark(
    *,
    dim: int = 128,
    n_patches: int = 1030,
    iters: int = 200,
    hidden: int = 256,
    depth: int = 2,
    device: str = "cpu",
    seed: int = 0,
    warmup: int = 20,
) -> dict:
    """Time P's per-patch forward and tally its storage / query costs. Returns the payload (no fingerprint).

    ``n_patches`` is one page's worth of patches (ColPali emits ~1030). We time a full-page forward
    ``iters`` times and take the median to reject scheduler noise, then divide by ``n_patches`` for the
    per-patch figure. All heavy imports are deferred so ``--help`` works without torch.
    """
    import torch

    from patchguard.defense.redact import RedactionProjection

    torch.manual_seed(seed)
    P = RedactionProjection(dim=dim, hidden=hidden, depth=depth).to(device).eval()

    param_count = int(sum(p.numel() for p in P.parameters()))
    elem_bytes = next(P.parameters()).element_size() if param_count else 4
    param_bytes = param_count * int(elem_bytes)

    # One page of patches on the unit sphere (matches the stored-embedding distribution shape).
    x = torch.randn(n_patches, dim, device=device)
    x = torch.nn.functional.normalize(x, dim=-1)

    with torch.no_grad():
        for _ in range(max(1, warmup)):  # warm caches / lazy allocs so timing is steady-state
            P(x)
        per_page_secs: list[float] = []
        for _ in range(max(1, iters)):
            t0 = time.perf_counter()
            P(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            per_page_secs.append(time.perf_counter() - t0)

    per_page = sorted(per_page_secs)
    median_page = per_page[len(per_page) // 2]
    us_per_patch = (median_page / n_patches) * 1e6
    ms_per_page = median_page * 1e3
    throughput = n_patches / median_page if median_page > 0 else float("inf")

    return {
        "mode": "efficiency_bench",
        "device": device,
        "dim": dim,
        "hidden": hidden,
        "depth": depth,
        "n_patches": n_patches,
        "iters": iters,
        "warmup": warmup,
        # (1) index-time latency — the only real cost
        "latency_us_per_patch": float(us_per_patch),
        "latency_ms_per_page": float(ms_per_page),
        "throughput_patches_per_sec": float(throughput),
        # (2) parameter storage — one-time model artifact, amortizes to ~0/page
        "param_count": param_count,
        "param_bytes": int(param_bytes),
        "param_bytes_amortized_per_page_at_1e6_pages": float(param_bytes / 1_000_000.0),
        # (3) per-page storage delta — P: d->d, so identical size
        "added_storage_bytes_per_page": 0,
        # (4) query-time overhead — queries never see P
        "query_time_overhead_us": 0.0,
        "query_added_bytes": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/efficiency_bench")
    ap.add_argument("--dim", type=int, default=128, help="embedding dim d (128 for ColPali)")
    ap.add_argument("--n-patches", type=int, default=1030, help="patches per page (ColPali ~1030)")
    ap.add_argument("--iters", type=int, default=200, help="timed full-page forwards (median reported)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="cpu|cuda — CPU is the honest index-builder cost")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from patchguard.repro import run_fingerprint, seed_everything
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    device = args.device
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda requested but unavailable; falling back to cpu")
        device = "cpu"

    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    payload = benchmark(dim=args.dim, n_patches=args.n_patches, iters=args.iters, hidden=args.hidden,
                        depth=args.depth, device=device, seed=args.seed, warmup=args.warmup)
    payload["fingerprint"] = run_fingerprint()

    print("=== RedactionProjection efficiency (P: d->d residual-MLP) ===")
    print(f"  device            : {payload['device']}")
    print(f"  P params          : {payload['param_count']:,} ({payload['param_bytes']:,} bytes, one-time)")
    print(f"  INDEX latency     : {payload['latency_us_per_patch']:.3f} µs/patch  "
          f"({payload['latency_ms_per_page']:.3f} ms/page @ {payload['n_patches']} patches)")
    print(f"  throughput        : {payload['throughput_patches_per_sec']:,.0f} patches/s")
    print(f"  per-page storage Δ: {payload['added_storage_bytes_per_page']} bytes (P is applied, not stored)")
    print(f"  query overhead    : {payload['query_time_overhead_us']} µs, {payload['query_added_bytes']} bytes "
          f"(queries use the vanilla encoder — P never runs on the query path)")

    (local_out / "efficiency_bench.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote efficiency_bench.json -> {args.out}")


if __name__ == "__main__":
    main()
