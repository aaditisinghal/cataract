"""RedactionProjection — a learned, index-time defense against the retrieval attack (MASTER_PLAN S8).

Our findings say a defense must be (a) in the stored embedding, (b) global (holographic bleed), and
(c) can't just cut capacity. Flat noise is isotropic — it degrades PII-similarity and content-similarity
EQUALLY. The hypothesis behind this module: PII-value queries and content queries occupy DIFFERENT
directions in ColPali's embedding space, so an ANISOTROPIC learned transform P applied to every stored
patch can suppress the PII directions while preserving the content directions — beating flat noise.

Threat model preserved: queries use the vanilla frozen encoder; only the STORED patches become P(patch).
So both retrieval and attack run MaxSim(vanilla_query, P(patch)). P is trained min-max:
  * utility  : content/topic query still ranks its own doc first (InfoNCE over a batch)
  * privacy  : the true PII value's MaxSim is pushed BELOW its distractors (un-rankable)
Sweep the privacy weight lambda to trace the frontier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RedactionProjection(nn.Module):
    """Per-patch residual transform, L2-normalized output (patches live on the unit sphere)."""

    def __init__(self, dim: int = 128, hidden: int = 256, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, dim)]
        self.mlp = nn.Sequential(*layers)
        self.gate = nn.Parameter(torch.tensor(0.1))  # start near identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.gate * self.mlp(x)
        return F.normalize(y, dim=-1)


def maxsim_batch(query: torch.Tensor, docs: torch.Tensor) -> torch.Tensor:
    """MaxSim of one query (nq,d) against a batch of docs (B,np,d) -> (B,) scores. Differentiable."""
    sims = torch.einsum("qd,bpd->bqp", query, docs)  # (B, nq, np)
    return sims.max(dim=2).values.sum(dim=1)  # max over patches, sum over query tokens


def train_redactor(
    patches: torch.Tensor,          # (B, np, d) train docs (stacked, fixed np)
    topic_q: list[torch.Tensor],    # per-doc content/topic query (nq,d)
    name_q: list[torch.Tensor],     # per-doc TRUE pii-value query (nq,d)
    distractor_q: torch.Tensor,     # (D, nq_pad, d)? -> we pass a python list instead
    lam: float,
    dim: int = 128,
    epochs: int = 400,
    lr: float = 1e-3,
    margin: float = 0.5,
    device: str = "cpu",
    seed: int = 0,
    distractors: list[torch.Tensor] | None = None,
) -> RedactionProjection:
    """Min-max train P. utility = InfoNCE(topic ranks own doc); privacy = push true-name below distractors."""
    torch.manual_seed(seed)
    P = RedactionProjection(dim=dim).to(device)
    opt = torch.optim.Adam(P.parameters(), lr=lr)
    patches = patches.to(device)
    B = patches.shape[0]
    topic_q = [t.to(device) for t in topic_q]
    name_q = [t.to(device) for t in name_q]
    distractors = [d.to(device) for d in (distractors or [])]

    for _ in range(epochs):
        Pt = P(patches)  # (B, np, d)
        # utility: score matrix S[i,j] = MaxSim(topic_i, doc_j); target diagonal
        rows = []
        for i in range(B):
            rows.append(maxsim_batch(topic_q[i], Pt))  # (B,)
        S = torch.stack(rows)  # (B,B)
        util = F.cross_entropy(S, torch.arange(B, device=device))

        # privacy: for each doc, true name score below its top distractor by margin
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
