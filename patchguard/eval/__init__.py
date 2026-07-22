"""Evaluation harness — the ruler, built before the attack (MASTER_PLAN S4)."""

from patchguard.eval.frontier import (
    bootstrap_ci,
    dominance_at,
    frontier_auc,
    two_proportion_z,
)
from patchguard.eval.killgate import GO, MEASUREMENT, STOP, KillGateResult, kill_gate
from patchguard.eval.pfrr import DEFAULT_CONFUSIONS, FieldResult, field_recovery, normalize, pfrr

__all__ = [
    "bootstrap_ci",
    "dominance_at",
    "frontier_auc",
    "two_proportion_z",
    "DEFAULT_CONFUSIONS",
    "FieldResult",
    "field_recovery",
    "normalize",
    "pfrr",
    "kill_gate",
    "KillGateResult",
    "GO",
    "MEASUREMENT",
    "STOP",
]
