"""The go/no-go gate (MASTER_PLAN ★). NOT YET RUNNABLE — needs S2 backends + S5 attack v0.

This file encodes the *contract* now so the thresholds are pre-registered in git with a commit
timestamp before the run (RESEARCH_PROTOCOL S4). Filling in the body happens at the ★ stage; the
numbers below must not change after real data is seen.

Design (exactly this, nothing more):
  200 FUNSD docs · ColPali vs BiPali at matched bytes/page · attack v0 only ·
  patch-scoped vs flat Gaussian × 6 noise levels · 5 seeds.

Gate (BOTH must hold):
  1. PFRR delta (ColPali - BiPali) >= 15pp
  2. frontier-AUC difference (patch-scoped - flat) excludes 0 at 95% (two_proportion_z / bootstrap)
"""

from __future__ import annotations

# --- PRE-REGISTERED THRESHOLDS (do not edit after the run; git history is the witness) ---
PFRR_DELTA_MIN_PP = 15.0  # percentage points, ColPali - BiPali
FRONTIER_AUC_CI_MUST_EXCLUDE_ZERO = True
N_DOCS = 200
N_SEEDS = 5
NOISE_LEVELS = 6
# -----------------------------------------------------------------------------------------


def main() -> None:
    raise NotImplementedError(
        "Kill test is pre-registered but not yet wired. Blocked on: "
        "retrievers/colpali.py + retrievers/bipali.py (S2) and attack/decoder.py (S5). "
        "See docs/MASTER_PLAN.md — do not implement S6+ before this passes."
    )


if __name__ == "__main__":
    main()
