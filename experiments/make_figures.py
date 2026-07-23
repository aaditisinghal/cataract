"""Render every paper figure from the bucketed result JSONs (COMPLETION_PLAN D3).

Single source of truth for the figures: point ``--runs`` at the local ``results/`` tree AND/OR the
``gs://patchguard-reakon-artifacts/runs`` prefix, and this script reads each experiment's ``*.json`` and
draws the seven paper figures as PDF (paper-ready, vector) + PNG (preview) into ``--figdir``.

Figures (each skips with a warning if its input JSON is absent, so a partial run still produces what it
can):
  fig1  recovered-PII bar          <- retrieval.json (name/id/dob top-1 & top-5 vs chance)
  fig2  Claim-1 + matched-bytes     <- claim1.json (ColPali vs BiPali) + claim1b.json (K-sweep)
  fig3  wrong-page 2x2 control       <- control_wrongpage.json (correct/wrong x full/erased)
  fig4  erasure dilation sweep       <- erasure.json (recovery vs deletion radius)
  fig5  cross-model panel            <- cross_model.json (leak / wrong-page / bleed, per backbone)
  fig6  glyph-height recovery curve  <- property_curve.json (+ funsd_transfer.json real-doc reference)
  fig7  THE defense frontier         <- learned_defense.json / baseline_frontier.json / defense_frontier.json
                                        / adaptive_attack.json (learned P vs flat noise vs baselines vs adaptive)

matplotlib is imported lazily inside ``build_figures`` so ``--help`` works on a box without it; if it is
missing, every figure is skipped cleanly (empty result, no crash). Multi-seed runs are handled by
averaging the scalar metrics across every matching JSON.
"""

from __future__ import annotations

import argparse
import glob
import json
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

_CHANCE_C = "#888888"


def _resolve_run_files(runs: str, tmp: Path) -> list[Path]:
    """Resolve one ``--runs`` token (gs:// prefix | glob | dir | single file) to sorted JSON paths."""
    s = str(runs).strip()
    if not s:
        return []
    if s.startswith("gs://"):
        from experiments.train_funsd import _gcs_download

        dest = Path(tempfile.mkdtemp(dir=tmp))
        local = _gcs_download(s, dest)
        return sorted(local.rglob("*.json"))
    if any(ch in s for ch in "*?[]"):
        return sorted(Path(p) for p in glob.glob(s, recursive=True))
    p = Path(s)
    if p.is_dir():
        return sorted(p.rglob("*.json"))
    return [p] if p.exists() else []


def _load_runs(runs: str) -> dict[str, list[dict]]:
    """Map each JSON basename (e.g. ``retrieval.json``) -> list of payloads found (one per seed/run)."""
    tmp = Path(tempfile.mkdtemp())
    files: list[Path] = []
    for tok in str(runs).split(","):
        files.extend(_resolve_run_files(tok, tmp))
    out: dict[str, list[dict]] = {}
    for f in files:
        try:
            payload = json.loads(Path(f).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out.setdefault(Path(f).name, []).append(payload)
    return out


def _all(runs: dict[str, list[dict]], name: str) -> list[dict]:
    return runs.get(name, [])


def _one(runs: dict[str, list[dict]], name: str) -> dict | None:
    lst = runs.get(name)
    return lst[0] if lst else None


def _mean(vals: list[float]) -> float:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else float("nan")


def _save(fig, figdir: Path, name: str) -> list[str]:
    """Write ``name.pdf`` + ``name.png`` into figdir; return the written paths as strings."""
    figdir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        p = figdir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        written.append(str(p))
    import matplotlib.pyplot as plt

    plt.close(fig)
    return written


# --------------------------------------------------------------------------------------------------
# individual figures — each returns a list of written paths, or [] if its input is missing/failed
# --------------------------------------------------------------------------------------------------
def fig1_recovered_pii(runs, figdir, plt) -> list[str]:
    payloads = _all(runs, "retrieval.json")
    if not payloads:
        warnings.warn("fig1: no retrieval.json — skipped")
        return []
    fields = ["name", "id_no", "dob"]
    top1 = [_mean([p.get("summary", {}).get(ft, {}).get("top1_acc") for p in payloads]) for ft in fields]
    top5 = [_mean([p.get("summary", {}).get(ft, {}).get("top5_acc") for p in payloads]) for ft in fields]
    chance = [payloads[0].get("summary", {}).get(ft, {}).get("chance", 0.0) for ft in fields]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(fields))
    w = 0.38
    ax.bar(x - w / 2, top1, w, label="top-1", color="#c0392b")
    ax.bar(x + w / 2, top5, w, label="top-5", color="#e59866")
    for xi, ch in zip(x, chance):
        ax.hlines(ch, xi - 0.45, xi + 0.45, color=_CHANCE_C, linestyle="--", linewidth=1.2)
    for xi, v in zip(x - w / 2, top1):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, ["name", "id no.", "dob"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recovery accuracy")
    ax.set_title(f"Retrieval / dictionary attack: recovered PII (n={len(payloads)} seed(s))")
    ax.text(0.99, 0.02, "-- chance", transform=ax.transAxes, ha="right", va="bottom",
            color=_CHANCE_C, fontsize=8)
    ax.legend(loc="upper right")
    return _save(fig, figdir, "fig1_recovered_pii")


def fig2_claim1(runs, figdir, plt) -> list[str]:
    c1 = _one(runs, "claim1.json")
    c1b = _one(runs, "claim1b.json")
    if c1 is None and c1b is None:
        warnings.warn("fig2: no claim1.json or claim1b.json — skipped")
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    if c1 is not None:
        fields = ["name", "id_no", "dob"]
        by = c1.get("summary", {}).get("by_field", {})
        cp = [by.get(ft, {}).get("colpali", {}).get("top1", np.nan) for ft in fields]
        bp = [by.get(ft, {}).get("bipali", {}).get("top1", np.nan) for ft in fields]

        def _err(side):
            e = []
            for ft in fields:
                ci = by.get(ft, {}).get(side, {}).get("ci", None)
                m = by.get(ft, {}).get(side, {}).get("top1", np.nan)
                if ci and len(ci) == 2 and not np.isnan(m):
                    e.append([m - ci[0], ci[1] - m])
                else:
                    e.append([0, 0])
            return np.array(e).T
        x = np.arange(len(fields))
        w = 0.38
        ax.bar(x - w / 2, cp, w, yerr=_err("colpali"), capsize=3, label="ColPali (multi-vector)",
               color="#2471a3")
        ax.bar(x + w / 2, bp, w, yerr=_err("bipali"), capsize=3, label="BiPali (mean-pooled)",
               color="#aed6f1")
        chance = by.get("name", {}).get("chance", 0.0)
        ax.axhline(chance, color=_CHANCE_C, linestyle="--", linewidth=1.2, label="chance")
        ax.set_xticks(x, ["name", "id no.", "dob"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("top-1 recovery")
        ax.set_title("Claim 1: leak is the aggregation, not the backbone")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "claim1.json missing", ha="center", va="center")

    ax = axes[1]
    if c1b is not None:
        sweep = c1b.get("sweep", [])
        ks = [s["k_patches"] for s in sweep]
        rec = [s["recovery"] for s in sweep]
        lo = [s.get("ci", [r, r])[0] for s, r in zip(sweep, rec)]
        hi = [s.get("ci", [r, r])[1] for s, r in zip(sweep, rec)]
        ax.plot(ks, rec, "-o", color="#2471a3", label="ColPali (K late-interaction patches)")
        ax.fill_between(ks, lo, hi, color="#2471a3", alpha=0.15)
        bp = c1b.get("bipali", {}).get("recovery", None)
        if bp is not None:
            ax.axhline(bp, color="#e67e22", linestyle="-.", linewidth=1.5,
                       label="BiPali (128 floats, pooled)")
        ax.axhline(c1b.get("chance", 0.0), color=_CHANCE_C, linestyle="--", linewidth=1.2, label="chance")
        mb = c1b.get("matched_bytes", {})
        if mb:
            ax.annotate(f"matched 128 floats:\nColPali-1 {mb.get('colpali_k1_128floats', float('nan')):.2f}"
                        f" vs BiPali {mb.get('bipali_128floats', float('nan')):.2f}",
                        xy=(ks[0], rec[0]), xytext=(0.28, 0.35), textcoords="axes fraction", fontsize=8,
                        arrowprops=dict(arrowstyle="->", color="#555"))
        if ks and min(ks) > 0:
            ax.set_xscale("log", base=2)
        ax.set_xlabel("K patches kept  (K x 128 floats)")
        ax.set_ylabel("name recovery")
        ax.set_ylim(0, 1.05)
        ax.set_title("Matched-bytes control: even 1 patch beats pooled")
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "claim1b.json missing", ha="center", va="center")

    fig.tight_layout()
    return _save(fig, figdir, "fig2_claim1_bipali_matchedbytes")


def fig3_wrongpage(runs, figdir, plt) -> list[str]:
    w = _one(runs, "control_wrongpage.json")
    if w is None:
        warnings.warn("fig3: no control_wrongpage.json — skipped")
        return []
    # rows: correct page / wrong page ; cols: full / name-erased
    M = np.array([[w.get("correct_full", np.nan), w.get("correct_erased", np.nan)],
                  [w.get("wrong_full", np.nan), w.get("wrong_erased", np.nan)]], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1], ["full page", "name-erased"])
    ax.set_yticks([0, 1], ["correct page\n(name present)", "wrong page\n(name absent)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=14,
                    color="black", fontweight="bold")
    ax.set_title(f"Wrong-page control (chance={w.get('chance', 0.0):.3f})\n{w.get('verdict', '')}",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="name recovery")
    fig.tight_layout()
    return _save(fig, figdir, "fig3_wrongpage_2x2")


def fig4_erasure(runs, figdir, plt) -> list[str]:
    e = _one(runs, "erasure.json")
    if e is None:
        warnings.warn("fig4: no erasure.json — skipped")
        return []
    sweep = e.get("sweep", [])
    if not sweep:
        warnings.warn("fig4: erasure.json has no sweep — skipped")
        return []
    r = [s["radius"] for s in sweep]
    rec = [s["recovery"] for s in sweep]
    lo = [s.get("ci", [x, x])[0] for s, x in zip(sweep, rec)]
    hi = [s.get("ci", [x, x])[1] for s, x in zip(sweep, rec)]
    removed = [s.get("patches_removed", np.nan) for s in sweep]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(r, rec, "-o", color="#8e44ad", label="name recovery")
    ax.fill_between(r, lo, hi, color="#8e44ad", alpha=0.15)
    wf = e.get("locality", {}).get("without_field", {}).get("acc", None)
    if wf is not None:
        ax.axhline(wf, color="#c0392b", linestyle=":", linewidth=1.5,
                   label=f"page minus field's own patches ({wf:.2f})")
    er = e.get("erasure_radius", None)
    if er is not None:
        ax.axvline(er, color="#27ae60", linestyle="--", linewidth=1.5, label=f"erasure radius r={er}")
    ax.set_xlabel("dilation radius r (grid neighbours deleted)")
    ax.set_ylabel("name recovery")
    ax.set_ylim(0, 1.05)
    ax.set_title("Erasure: how far you must delete before the name is gone")

    ax2 = ax.twinx()
    ax2.plot(r, removed, "-s", color="#95a5a6", alpha=0.7, markersize=4, label="patches removed")
    ax2.set_ylabel("avg patches removed", color="#7f8c8d")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
    fig.tight_layout()
    return _save(fig, figdir, "fig4_erasure_dilation")


def fig5_crossmodel(runs, figdir, plt) -> list[str]:
    payloads = _all(runs, "cross_model.json")
    if not payloads:
        warnings.warn("fig5: no cross_model.json — skipped")
        return []
    # group by retriever, averaging across seeds
    by_ret: dict[str, list[dict]] = {}
    for p in payloads:
        by_ret.setdefault(p.get("retriever", "?"), []).append(p)
    metrics = ["leak", "wrong_page", "bleed50", "bleed90"]
    labels = ["leak\n(correct)", "wrong-page", "bleed\n-50% patches", "bleed\n-90% patches"]

    def _val(p, m):
        if m == "leak":
            return p.get("leak", np.nan)
        if m == "wrong_page":
            return p.get("wrong_page", np.nan)
        bl = p.get("bleed", {})
        key = "0.5" if m == "bleed50" else "0.9"
        return bl.get(key, np.nan)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    rets = sorted(by_ret)
    x = np.arange(len(metrics))
    w = 0.8 / max(1, len(rets))
    colors = ["#16a085", "#2980b9", "#8e44ad", "#d35400"]
    for i, ret in enumerate(rets):
        vals = [_mean([_val(p, m) for p in by_ret[ret]]) for m in metrics]
        ax.bar(x + (i - (len(rets) - 1) / 2) * w, vals, w, label=ret, color=colors[i % len(colors)])
    chance = payloads[0].get("chance", 0.0)
    ax.axhline(chance, color=_CHANCE_C, linestyle="--", linewidth=1.2, label=f"chance ({chance:.3f})")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("name recovery")
    ax.set_title("Cross-model: leak + holographic bleed generalise across multi-vector backbones")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return _save(fig, figdir, "fig5_crossmodel_panel")


def fig6_glyph_curve(runs, figdir, plt) -> list[str]:
    pc = _one(runs, "property_curve.json")
    if pc is None:
        warnings.warn("fig6: no property_curve.json — skipped")
        return []
    curve = pc.get("curve", [])
    if not curve:
        warnings.warn("fig6: property_curve.json has no curve — skipped")
        return []
    gh = [pt["glyph_px"] for pt in curve]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    colors = {"name": "#c0392b", "id_no": "#2471a3", "dob": "#27ae60"}
    for ft, lab in (("name", "name"), ("id_no", "id no."), ("dob", "dob")):
        y = [pt.get(ft, {}).get("top1", np.nan) for pt in curve]
        ax.plot(gh, y, "-o", color=colors[ft], label=f"synthetic {lab}")
    # real-doc reference from FUNSD, if present
    fn = _one(runs, "funsd_transfer.json")
    if fn is not None:
        summ = fn.get("summary", {})
        t1 = summ.get("top1_acc", None)
        if t1 is not None:
            ax.axhline(t1, color="#7f8c8d", linestyle="-.", linewidth=1.4,
                       label=f"FUNSD real-doc overall ({t1:.2f})")
        ch = summ.get("chance", None)
        if ch is not None:
            ax.axhline(ch, color=_CHANCE_C, linestyle="--", linewidth=1.0,
                       label=f"FUNSD chance ({ch:.3f})")
    ax.axhline(0.5, color="#bbbbbb", linestyle=":", linewidth=1.0)
    ax.set_xlabel("glyph height (px)")
    ax.set_ylabel("top-1 recovery")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recovery scales with glyph height (all corpora on one axis)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _save(fig, figdir, "fig6_glyph_height_curve")


def _frontier_xy(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Sort (privacy, utility) points by privacy for a clean monotone-ish curve."""
    if not points:
        return np.array([]), np.array([])
    pr = np.array([p.get("privacy", np.nan) for p in points], dtype=float)
    ut = np.array([p.get("utility", np.nan) for p in points], dtype=float)
    order = np.argsort(pr)
    return pr[order], ut[order]


def fig7_defense_frontier(runs, figdir, plt) -> list[str]:
    bf = _one(runs, "baseline_frontier.json")
    ld = _one(runs, "learned_defense.json")
    df = _one(runs, "defense_frontier.json")
    aa = _one(runs, "adaptive_attack.json")
    if bf is None and ld is None and df is None:
        warnings.warn("fig7: no defense frontier JSONs — skipped")
        return []
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # learned P (prefer the baselines run's learned frontier, else learned_defense's)
    learned = (bf or {}).get("learned_frontier") or (ld or {}).get("learned_frontier") or []
    if learned:
        px, uy = _frontier_xy(learned)
        ax.plot(px, uy, "-o", color="#1a5276", linewidth=2.4, markersize=6, zorder=5,
                label="learned P (ours)")

    # flat noise baseline
    flat = (ld or {}).get("flat_frontier") or []
    if not flat and df is not None:
        flat = df.get("frontier", {}).get("flat", [])
    if flat:
        px, uy = _frontier_xy(flat)
        ax.plot(px, uy, "--s", color="#c0392b", label="flat Gaussian noise")

    # patch-scoped (the "local defense is impossible" curve)
    scoped = (df or {}).get("frontier", {}).get("patch_scoped", [])
    if scoped:
        px, uy = _frontier_xy(scoped)
        ax.plot(px, uy, ":d", color="#e67e22", label="patch-scoped noise")

    # ported baselines
    base_colors = {"entroguard": "#8e44ad", "press": "#16a085", "koga": "#7f8c8d"}
    for name, pts in ((bf or {}).get("baseline_frontiers", {}) or {}).items():
        px, uy = _frontier_xy(pts)
        if px.size:
            ax.plot(px, uy, "-^", color=base_colors.get(name, "#555"), alpha=0.8, markersize=4,
                    label=f"{name}")

    # adaptive attacker: the privacy an adaptive (white-box) attacker actually leaves standing
    if aa is not None:
        strat = aa.get("strategies", {})
        if strat:
            best_rec = max((v.get("recovery", 0.0) for v in strat.values()), default=0.0)
            adaptive_priv = 1.0 - best_rec
            ax.axvline(adaptive_priv, color="#8b0000", linestyle="-.", linewidth=1.6,
                       label=f"adaptive attacker (privacy≈{adaptive_priv:.2f})")
            # place a marker at the operating-lambda utility if we can find it
            lam = aa.get("lam")
            util_at_lam = None
            for p in learned:
                if lam is not None and abs(float(p.get("lambda", -1)) - float(lam)) < 1e-9:
                    util_at_lam = p.get("utility")
            if util_at_lam is not None:
                ax.scatter([adaptive_priv], [util_at_lam], marker="X", s=110, color="#8b0000",
                           zorder=6)
        na = aa.get("baselines", {}).get("non_adaptive_privacy")
        if na is not None:
            ax.axvline(na, color="#1a5276", linestyle=":", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("privacy  (1 - name recovery)")
    ax.set_ylabel("utility  (topic retrieval)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_title("The defense frontier: learned P vs flat noise vs baselines vs adaptive attacker")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _save(fig, figdir, "fig7_defense_frontier")


_FIGS = [fig1_recovered_pii, fig2_claim1, fig3_wrongpage, fig4_erasure,
         fig5_crossmodel, fig6_glyph_curve, fig7_defense_frontier]


def build_figures(runs: str, figdir: str) -> list[str]:
    """Load the run JSONs, render every figure, and return the list of written file paths.

    matplotlib is imported here (lazily): if it is unavailable, all figures are skipped and ``[]`` is
    returned without raising, so the smoke test passes on a matplotlib-free box.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib absent -> clean skip, no crash
        warnings.warn(f"matplotlib unavailable ({exc!r}); skipping all figures")
        return []

    runs_data = _load_runs(runs)
    fdir = Path(figdir)
    written: list[str] = []
    for fn in _FIGS:
        try:
            written.extend(fn(runs_data, fdir, plt))
        except Exception as exc:  # one bad figure must not kill the rest
            warnings.warn(f"{fn.__name__} failed: {exc!r}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results",
                    help="comma-separated dir(s) / glob(s) / gs:// prefix(es) holding result JSONs")
    ap.add_argument("--figdir", default="paper/figures", help="output dir for PDF+PNG figures")
    args = ap.parse_args()

    written = build_figures(args.runs, args.figdir)
    if not written:
        print("no figures written (matplotlib missing or no input JSONs found)")
        return
    print(f"wrote {len(written)} files into {args.figdir}:")
    seen = set()
    for p in written:
        stem = Path(p).stem
        if stem not in seen:
            seen.add(stem)
            print(f"  - {stem} (.pdf + .png)")


if __name__ == "__main__":
    main()
