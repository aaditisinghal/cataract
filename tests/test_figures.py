"""CPU tests for the figures + efficiency-bench deliverables (COMPLETION_PLAN C1/D3).

No ColPali, no GPU, no network. We build a tiny fake ``runs/`` tree of result JSONs shaped like the
real experiment outputs and assert ``make_figures.build_figures`` either writes files (matplotlib
present) or cleanly returns ``[]`` (matplotlib absent) — and that it never crashes on missing inputs.
Then we exercise ``efficiency_bench`` (torch-only, so it runs here) and assert the payload shape.
"""

import json
from pathlib import Path

import pytest

from experiments import make_figures
from experiments.efficiency_bench import benchmark


def _has_mpl() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def _write(d: Path, name: str, payload: dict) -> None:
    sub = d / name.replace(".json", "-tag")
    sub.mkdir(parents=True, exist_ok=True)
    (sub / name).write_text(json.dumps(payload))


def _fake_runs(root: Path) -> Path:
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    _write(runs, "retrieval.json", {
        "mode": "retrieval_attack",
        "summary": {
            "name": {"top1_acc": 1.0, "top5_acc": 1.0, "chance": 1 / 240},
            "id_no": {"top1_acc": 0.72, "top5_acc": 0.9, "chance": 1 / 1000},
            "dob": {"top1_acc": 0.40, "top5_acc": 0.7, "chance": 1 / 1000},
        }})
    _write(runs, "claim1.json", {
        "mode": "claim1",
        "summary": {"by_field": {
            "name": {"colpali": {"top1": 1.0, "ci": [0.9, 1.0]},
                     "bipali": {"top1": 0.1, "ci": [0.05, 0.2]}, "delta": 0.9, "chance": 1 / 201},
            "id_no": {"colpali": {"top1": 0.7, "ci": [0.6, 0.8]},
                      "bipali": {"top1": 0.05, "ci": [0.0, 0.1]}, "delta": 0.65, "chance": 1 / 201},
            "dob": {"colpali": {"top1": 0.4, "ci": [0.3, 0.5]},
                    "bipali": {"top1": 0.02, "ci": [0.0, 0.05]}, "delta": 0.38, "chance": 1 / 201},
        }}})
    _write(runs, "claim1b.json", {
        "mode": "claim1b",
        "sweep": [{"k_patches": k, "floats": k * 128, "recovery": min(1.0, 0.5 + 0.05 * k),
                   "ci": [0.4, 0.9]} for k in (1, 2, 4, 8, 16, 64, 256)],
        "bipali": {"recovery": 0.1, "ci": [0.05, 0.2], "floats": 128},
        "matched_bytes": {"colpali_k1_128floats": 0.55, "bipali_128floats": 0.1,
                          "delta_at_matched_bytes": 0.45},
        "chance": 1 / 201})
    _write(runs, "control_wrongpage.json", {
        "mode": "control_wrongpage", "chance": 1 / 201,
        "correct_full": 1.0, "wrong_full": 0.02, "correct_erased": 0.9, "wrong_erased": 0.02,
        "verdict": "HOLOGRAPHIC BLEED CONFIRMED"})
    _write(runs, "erasure.json", {
        "mode": "erasure",
        "locality": {"without_field": {"acc": 0.86, "ci": [0.7, 0.95]}},
        "sweep": [{"radius": r, "recovery": max(0.0, 0.9 - 0.12 * r), "ci": [0.0, 1.0],
                   "patches_removed": 4 * (r + 1)} for r in (0, 1, 2, 3, 4, 6, 8)],
        "erasure_radius": 6})
    _write(runs, "cross_model.json", {
        "mode": "cross_model", "retriever": "colqwen2", "chance": 1 / 201,
        "leak": 0.95, "wrong_page": 0.03, "bleed": {"0.25": 0.9, "0.5": 0.8, "0.75": 0.6, "0.9": 0.5}})
    _write(runs, "property_curve.json", {
        "mode": "property_curve",
        "curve": [{"font": f, "glyph_px": f, "name": {"top1": min(1.0, f / 40)},
                   "id_no": {"top1": min(1.0, f / 60)}, "dob": {"top1": min(1.0, f / 80)}}
                  for f in (8, 12, 16, 24, 32, 40)]})
    _write(runs, "funsd_transfer.json", {
        "mode": "funsd_transfer",
        "summary": {"top1_acc": 0.26, "top5_acc": 0.5, "chance": 1 / 20}})
    _write(runs, "learned_defense.json", {
        "mode": "learned_defense",
        "learned_frontier": [{"lambda": lam, "utility": 0.95 - 0.02 * i, "privacy": 0.1 * (i + 1)}
                             for i, lam in enumerate([0, 1, 2, 5, 10])],
        "flat_frontier": [{"noise": s, "utility": 0.9 - 0.15 * i, "privacy": 0.1 + 0.18 * i}
                          for i, s in enumerate([0, 0.1, 0.2, 0.35, 0.5])]})
    _write(runs, "baseline_frontier.json", {
        "mode": "baseline_frontier",
        "learned_frontier": [{"lambda": lam, "utility": 0.95 - 0.02 * i, "privacy": 0.15 * (i + 1)}
                             for i, lam in enumerate([0, 2, 5, 10])],
        "baseline_frontiers": {
            "entroguard": [{"strength": s, "utility": 0.9 - 0.2 * i, "privacy": 0.1 + 0.2 * i}
                           for i, s in enumerate([0.05, 0.2, 0.5])],
            "press": [{"strength": s, "utility": 0.92 - 0.18 * i, "privacy": 0.12 + 0.19 * i}
                      for i, s in enumerate([0.05, 0.2, 0.5])],
            "koga": [{"strength": s, "utility": 0.88 - 0.22 * i, "privacy": 0.1 + 0.22 * i}
                     for i, s in enumerate([0.05, 0.2, 0.5])]}})
    _write(runs, "defense_frontier.json", {
        "mode": "defense_frontier",
        "frontier": {"flat": [{"noise": 0.1, "utility": 0.8, "privacy": 0.3}],
                     "patch_scoped": [{"noise": 0.1, "utility": 0.85, "privacy": 0.05},
                                      {"noise": 0.5, "utility": 0.4, "privacy": 0.1}]}})
    _write(runs, "adaptive_attack.json", {
        "mode": "adaptive_attack", "lam": 5.0,
        "baselines": {"non_adaptive_privacy": 0.98, "dict_chance": 1 / 201},
        "strategies": {
            "B1a_distillation": {"recovery": 0.6, "chance": 1 / 32},
            "B1b_inverse": {"recovery": 0.1, "chance": 1 / 201}}})
    return runs


def test_build_figures_writes_or_skips(tmp_path):
    runs = _fake_runs(tmp_path)
    figdir = tmp_path / "figs"
    written = make_figures.build_figures(str(runs), str(figdir))
    assert isinstance(written, list)
    if _has_mpl():
        assert written, "matplotlib present but no figures were written"
        for p in written:
            assert Path(p).exists() and Path(p).stat().st_size > 0
        # both a pdf and a png land for at least one figure
        exts = {Path(p).suffix for p in written}
        assert ".pdf" in exts and ".png" in exts
        # all seven figures should render from the complete fake run set
        stems = {Path(p).stem for p in written}
        assert len(stems) == 7
    else:
        assert written == []


def test_build_figures_missing_inputs_no_crash(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    # no JSONs at all -> every figure skips, no exception either way
    written = make_figures.build_figures(str(empty), str(tmp_path / "figs2"))
    assert written == []


def test_build_figures_partial_inputs(tmp_path):
    runs = tmp_path / "partial"
    runs.mkdir()
    _write(runs, "retrieval.json", {
        "mode": "retrieval_attack",
        "summary": {"name": {"top1_acc": 1.0, "top5_acc": 1.0, "chance": 1 / 240},
                    "id_no": {"top1_acc": 0.5, "top5_acc": 0.7, "chance": 1 / 1000},
                    "dob": {"top1_acc": 0.3, "top5_acc": 0.5, "chance": 1 / 1000}}})
    written = make_figures.build_figures(str(runs), str(tmp_path / "figs3"))
    if _has_mpl():
        stems = {Path(p).stem for p in written}
        assert stems == {"fig1_recovered_pii"}  # only the figure whose input exists
    else:
        assert written == []


def test_efficiency_bench_payload_keys():
    payload = benchmark(dim=16, n_patches=32, iters=20, hidden=32, depth=2, seed=0, warmup=3)
    expected = {
        "mode", "device", "dim", "hidden", "depth", "n_patches", "iters",
        "latency_us_per_patch", "latency_ms_per_page", "throughput_patches_per_sec",
        "param_count", "param_bytes", "added_storage_bytes_per_page", "query_time_overhead_us",
        "query_added_bytes",
    }
    assert expected <= set(payload)
    assert payload["mode"] == "efficiency_bench"
    assert payload["param_count"] > 0
    assert payload["latency_us_per_patch"] > 0.0
    assert payload["throughput_patches_per_sec"] > 0.0
    # the three "must be zero" costs
    assert payload["added_storage_bytes_per_page"] == 0
    assert payload["query_time_overhead_us"] == 0.0
    assert payload["query_added_bytes"] == 0


def test_efficiency_bench_main_writes_json(tmp_path, monkeypatch):
    import sys

    from experiments import efficiency_bench

    out = tmp_path / "eff"
    argv = ["efficiency_bench", "--out", str(out), "--dim", "16", "--n-patches", "32",
            "--iters", "10", "--warmup", "2", "--hidden", "32"]
    monkeypatch.setattr(sys, "argv", argv)
    efficiency_bench.main()
    payload = json.loads((out / "efficiency_bench.json").read_text())
    assert payload["mode"] == "efficiency_bench"
    assert "fingerprint" in payload
