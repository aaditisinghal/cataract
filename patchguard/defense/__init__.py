"""Defenses (MASTER_PLAN S7 baselines, S8 patch-scoped, S10 erasure)."""

from patchguard.defense.localize import OracleLocalizer
from patchguard.defense.perturb import flat_gaussian, patch_scoped_gaussian

__all__ = ["OracleLocalizer", "flat_gaussian", "patch_scoped_gaussian"]
