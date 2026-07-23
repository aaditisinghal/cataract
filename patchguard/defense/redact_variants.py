"""Alternative redaction architectures for the defense ABLATION (MASTER_PLAN S8, plan item B7).

``redact.py`` ships one learned index-time defense: a residual GELU-MLP with a learnable gate
(``RedactionProjection``). But the science question the ablation answers is *how much of that machinery
is load-bearing*. If a single **linear** residual — an interpretable, invertible-in-principle map — already
carves the PII directions out of the anisotropic embedding while sparing the content directions, then the
defense is not a black box: it is one learned matrix, and the paper can say so. Conversely, if privacy only
appears once depth/nonlinearity/gating are added, that tells us the PII manifold is not linearly separable
from the content manifold and the extra capacity is justified.

This module supplies the architectures the sweep needs while keeping the exact forward CONTRACT of
``RedactionProjection`` — input ``(..., d)`` -> L2-normalized ``(..., d)`` residual output, applied per
patch at index time, queries left on the frozen vanilla encoder:

  * ``LinearRedaction``       — ``y = x + gate * W x`` (no nonlinearity). The minimal, interpretable P.
  * ``build_redactor``        — factory for {linear, mlp-depth1..N} x {gate learned/fixed} x hidden size,
                                reusing ``RedactionProjection`` (imported, never edited) for the MLP cells.
  * ``train_variant``         — the SAME min-max objective as ``train_redactor`` (utility = InfoNCE topic
                                ranking, privacy = push the true name below its distractors by a margin),
                                but accepting a pre-built P of ANY architecture, since ``train_redactor``
                                hard-codes ``RedactionProjection``.

A positive ablation result ("linear suffices") is a strong, interpretable finding; a negative one
("depth/gating required") quantifies exactly what capacity the defense needs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from patchguard.defense.redact import RedactionProjection, maxsim_batch


def _parse_gate(gate: object) -> tuple[str, float | None]:
    """Normalize a gate config to ``('learned', None)`` or ``('fixed', value)``.

    Accepts: ``True`` / ``"learned"`` / ``"on"`` -> learnable scalar (init 0.1, as in RedactionProjection);
    ``False`` / ``"off"`` / ``"fixed"`` / ``"full"`` -> fixed full-strength residual (value 1.0);
    a number (or numeric string) -> fixed residual at that strength.
    """
    if gate is True:
        return ("learned", None)
    if gate is False:
        return ("fixed", 1.0)
    if isinstance(gate, str):
        g = gate.strip().lower()
        if g in ("learned", "on", "learn"):
            return ("learned", None)
        if g in ("off", "fixed", "full"):
            return ("fixed", 1.0)
        return ("fixed", float(g))
    return ("fixed", float(gate))


class LinearRedaction(nn.Module):
    """The minimal learned P: a single linear residual, L2-normalized. ``y = x + gate * W x``.

    No GELU, no hidden layer — the interpretable lower bound for the ablation. ``bias=False`` keeps it a
    pure linear map ``W x`` (matching the residual math and avoiding a constant per-patch offset). The gate
    is either a learnable scalar (starts near identity like ``RedactionProjection``) or a fixed constant.
    """

    def __init__(self, dim: int = 128, gate: object = "learned", bias: bool = False) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=bias)
        mode, val = _parse_gate(gate)
        if mode == "learned":
            self.gate = nn.Parameter(torch.tensor(0.1))
        else:
            # Fixed (non-trainable) gate: registered as a buffer so it moves with .to(device)
            # but is excluded from .parameters().
            self.register_buffer("gate", torch.tensor(float(val)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.gate * self.lin(x)
        return F.normalize(y, dim=-1)


def _apply_gate_config(P: nn.Module, gate: object) -> nn.Module:
    """Set the gate of an existing ``RedactionProjection`` (learned vs frozen at a fixed strength).

    We never edit ``redact.py``: to turn the gate "off" we freeze the existing scalar Parameter at the
    fixed value (``requires_grad=False``) so the residual runs at constant strength and is not learned.
    """
    mode, val = _parse_gate(gate)
    if mode == "learned":
        return P
    with torch.no_grad():
        P.gate.data.fill_(float(val))
    P.gate.requires_grad_(False)
    return P


def build_redactor(
    variant: str, dim: int = 128, hidden: int = 256, gate: object = "learned",
    depth: int | None = None,
) -> nn.Module:
    """Construct a redactor for an ablation cell.

    ``variant`` is one of:
      * ``"linear"`` (aka ``"mlp-depth0"``)  -> ``LinearRedaction``
      * ``"mlp-depth<k>"`` for k>=1           -> ``RedactionProjection(depth=k, hidden=hidden)``
      * ``"mlp"``                             -> ``RedactionProjection(depth=depth or 2, hidden=hidden)``
    ``gate`` follows ``_parse_gate`` (learned scalar vs fixed strength). Same forward contract for all.
    """
    v = variant.strip().lower()
    if v in ("linear", "mlp-depth0", "depth0"):
        return LinearRedaction(dim=dim, gate=gate)
    if v.startswith("mlp-depth") or v.startswith("depth"):
        d = int(v.split("depth")[-1])
        if d == 0:
            return LinearRedaction(dim=dim, gate=gate)
        return _apply_gate_config(RedactionProjection(dim=dim, hidden=hidden, depth=d), gate)
    if v == "mlp":
        d = depth if depth is not None else 2
        return _apply_gate_config(RedactionProjection(dim=dim, hidden=hidden, depth=d), gate)
    raise ValueError(f"unknown redactor variant {variant!r}")


def train_variant(
    P: nn.Module,
    patches: torch.Tensor,        # (B, np, d) train docs (stacked, fixed np)
    topic_q: list[torch.Tensor],  # per-doc content/topic query (nq, d)
    name_q: list[torch.Tensor],   # per-doc TRUE pii-value query (nq, d)
    distractors: list[torch.Tensor] | None,
    lam: float,
    epochs: int = 400,
    lr: float = 1e-3,
    margin: float = 0.5,
    device: str = "cpu",
) -> nn.Module:
    """Min-max train a PRE-BUILT redactor P of any architecture.

    Identical objective to ``patchguard.defense.redact.train_redactor`` (utility = InfoNCE topic ranking,
    privacy = hinge pushing the true name below its top distractor), but P is supplied by the caller so the
    ablation can train ``LinearRedaction`` and gate-frozen MLPs that ``train_redactor`` cannot construct.
    Frozen parameters (e.g. a fixed gate) are excluded from the optimizer.
    """
    P = P.to(device)
    trainable = [p for p in P.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr)
    patches = patches.to(device)
    B = patches.shape[0]
    topic_q = [t.to(device) for t in topic_q]
    name_q = [t.to(device) for t in name_q]
    distractors = [d.to(device) for d in (distractors or [])]

    for _ in range(epochs):
        Pt = P(patches)  # (B, np, d)
        rows = [maxsim_batch(topic_q[i], Pt) for i in range(B)]  # each (B,)
        S = torch.stack(rows)  # (B, B) — S[i,j] = MaxSim(topic_i, doc_j)
        util = F.cross_entropy(S, torch.arange(B, device=device))

        priv = torch.zeros((), device=device)
        if lam > 0 and distractors:
            for i in range(B):
                s_true = maxsim_batch(name_q[i], Pt[i : i + 1])[0]
                d_scores = torch.stack([maxsim_batch(dq, Pt[i : i + 1])[0] for dq in distractors])
                priv = priv + F.relu(s_true - d_scores.max() + margin)
            priv = priv / B

        loss = util + lam * priv
        opt.zero_grad()
        loss.backward()
        opt.step()
    return P.eval()
