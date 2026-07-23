#!/usr/bin/env bash
# reproduce.sh — regenerate EVERY result JSON, then all paper figures, in dependency order.
#
# Two run paths per experiment:
#   * WARM-VM fast path (default, ~3.5 min/exp): scripts/21_exec.sh <module> <out_subdir> [args...]
#       Uploads to gs://patchguard-reakon-artifacts/runs/<out_subdir>-<image_tag>/<name>.json.
#       Bring the warm VM up ONCE first:            scripts/20_warm_vm.sh us-central1-a a100
#       Tear it down when finished:                 gcloud compute instances delete "$(cat .warm_vm)" \
#                                                        --zone="$(cat .warm_zone)" --quiet
#   * COLD one-shot GCE alternative (self-deleting VM per job) — shown commented under each block:
#       scripts/12_train_vm.sh us-central1-a a100 <job>
#
# The two CPU-local steps (efficiency_bench, make_figures) need no GPU and run on this workstation.
# Run from anywhere: the script cd's to the repo root so scripts/*.sh and .warm_vm resolve.
set -euo pipefail
cd "$(dirname "$0")"

RUNS="gs://patchguard-reakon-artifacts/runs"
FUNSD="gs://patchguard-reakon-data/funsd"
SEEDS="${SEEDS:-0 1 2 3 4}"   # headline experiments run at 5 seeds (COMPLETION_PLAN A2 — Holm-Bonferroni)

exec_exp() { echo "### exec $*"; scripts/21_exec.sh "$@"; }

# ==================================================================================================
# PHASE 1 — ATTACK / FINDINGS
# ==================================================================================================

# --- headline attack experiments, 5 seeds each (aggregated below) -------------------------------
for s in $SEEDS; do
  exec_exp experiments.retrieval_attack  "retrieval-seed${s}"  --n 40 --distractors 999 --font-size 34 --seed "$s"
  exec_exp experiments.claim1            "claim1-seed${s}"     --n 60 --distractors 200 --font-size 24 --seed "$s"
  exec_exp experiments.control_wrongpage "wrongpage-seed${s}"  --n 50 --distractors 200 --font-size 24 --seed "$s"
  exec_exp experiments.erasure           "erasure-seed${s}"    --n 50 --distractors 200 --font-size 24 --seed "$s"
  exec_exp experiments.cross_model       "crossmodel-seed${s}" --retriever colqwen2 --n 40 --distractors 200 --font-size 24 --seed "$s"
done
# cold alternative (single seed): scripts/12_train_vm.sh us-central1-a a100 retrieval
#                                 scripts/12_train_vm.sh us-central1-a a100 claim1
#                                 scripts/12_train_vm.sh us-central1-a a100 wrongpage
#                                 scripts/12_train_vm.sh us-central1-a a100 erasure
#                                 scripts/12_train_vm.sh us-central1-a a100 crossmodel

# --- single-seed attack experiments -------------------------------------------------------------
exec_exp experiments.claim1b           "claim1b"      --n 50 --distractors 200 --font-size 24
exec_exp experiments.property_curve    "curve"        --n 30 --fonts 8,10,12,14,16,20,24,32,40,48 --distractors 200 --anagrams 8
exec_exp experiments.retrieval_rerank  "rerank"       --n 40 --distractors 999 --font-size 34 --topk 20   # A4 positional rerank
exec_exp experiments.arrangement_control "arrangement" --n 40 --distractors 200 --font-size 24            # A5 arrangement null
# FUNSD real-doc transfer: all labels + the fair answer-only number (A1)
exec_exp experiments.funsd_transfer    "funsd"        --data "$FUNSD" --n-pages 60 --k 20 --max-fields 12
exec_exp experiments.funsd_transfer    "funsd-answer" --data "$FUNSD" --labels answer --n-pages 60 --k 20 --max-fields 12
# cold alternative: scripts/12_train_vm.sh us-central1-a a100 claim1b
#                   scripts/12_train_vm.sh us-central1-a a100 curve
#                   scripts/12_train_vm.sh us-central1-a a100 funsd

# ==================================================================================================
# PHASE 2 — SOLUTION / DEFENSE
# ==================================================================================================

# --- learned defense frontier, 5 seeds (the privacy/utility headline) ---------------------------
for s in $SEEDS; do
  exec_exp experiments.learned_defense "defense2-seed${s}" --n-train 64 --n-test 40 --epochs 300 --seed "$s"
done
# cold alternative: scripts/12_train_vm.sh us-central1-a a100 defense2

# --- single-seed defense experiments ------------------------------------------------------------
exec_exp experiments.defense_frontier   "defense"            --n 50 --distractors 200 --font-size 24        # local-noise impossibility
exec_exp experiments.baseline_frontier  "baselines"          --n-train 64 --n-test 40 --lams 0,2,5,10 --epochs 300  # B5 EntroGuard/PRESS/Koga vs P
exec_exp experiments.adaptive_attack    "adaptive"           --lam 5.0 --n-train 64 --n-test 40 --epochs 300 # B1 THE decisive adaptive test
exec_exp experiments.defense_floor      "floor"              --alphas 0,0.25,0.5,0.75,1.0 --lams 0,1,5,20    # B3 entanglement floor
exec_exp experiments.defense_crossmodel "defense_crossmodel" --retriever colqwen2 --lams 0,2,5,10 --epochs 300 # B4 P on ColQwen2
exec_exp experiments.defense_ablation   "defense_ablation"   --lam 5.0 --epochs 300                          # B7 architecture ablation
exec_exp experiments.vidore_utility     "vidore"             --lam 5.0 --n-corpus 60 --n-queries 60 --epochs 300 # B2 real-retrieval NDCG@5
# vidore synthetic fallback (no ViDoRe download): add --synthetic to the line above.
# cold alternative: scripts/12_train_vm.sh us-central1-a a100 defense

# ==================================================================================================
# AGGREGATE the multi-seed families into mean +/- 95% CI + Holm-Bonferroni (A2)
# ==================================================================================================
exec_agg() { echo "### agg $*"; scripts/21_exec.sh experiments.aggregate_seeds "$@"; }
exec_agg "agg-retrieval"  --runs "$RUNS" --metric-keys summary.name.top1_acc,summary.id_no.top1_acc,summary.dob.top1_acc
exec_agg "agg-claim1"     --runs "$RUNS" --metric-keys summary.by_field.name.delta,summary.by_field.id_no.delta,summary.by_field.dob.delta
exec_agg "agg-wrongpage"  --runs "$RUNS" --metric-keys correct_full,wrong_full,correct_erased,wrong_erased
exec_agg "agg-erasure"    --runs "$RUNS" --metric-keys locality.without_field.acc
exec_agg "agg-crossmodel" --runs "$RUNS" --metric-keys leak,wrong_page
# NB: aggregate_seeds globs every JSON under --runs and keeps only files that contain the requested
# metric keys, so pointing each call at the whole runs prefix resolves each family cleanly.

# ==================================================================================================
# CPU-LOCAL steps (no GPU): deploy-cost table + all figures
# ==================================================================================================
echo "### local efficiency_bench (C1)"
python3 -m experiments.efficiency_bench --out results/efficiency_bench --n-patches 1030 --iters 200

echo "### local make_figures (D3) — reads every JSON above, writes paper/figures/*.{pdf,png}"
python3 -m experiments.make_figures --runs "$RUNS" --figdir paper/figures

echo "DONE. Figures in paper/figures/ ; per-experiment JSONs under ${RUNS}/."

# ==================================================================================================
# PHASE 3 — COMPLETION-PLAN ADDITIONS (appended; the base reproduction above is unchanged).
#   New experiments that finalize the attack breadth + the constructive (information-DESTROYING)
#   defense: B6 certified/nullspace defense + its provable bound, A3 real-corpus (CORD) transfer,
#   A6 ghost vectors, B8 real-doc defense transfer, A7 transferable (proxy-encoder) attack.
#   Because the make_figures tail above runs before this block, we RE-AGGREGATE + RE-DRAW figures at
#   the very end so the paper figures include every PHASE-3 JSON as well.
# ==================================================================================================

# --- B6 the constructive answer to §4.13: information-DESTROYING (nullspace) defense vs inverse attack
exec_exp experiments.certified_defense      "certified"        --ks 0,2,4,8,16,32 --n-train 64 --n-test 40
# --- A3 real-corpus transfer: the retrieval attack on CORD receipts (financial PII), glyph-height axis
exec_exp experiments.realcorpus_transfer    "realcorpus-cord"  --corpus cord --n-pages 60 --k 20
# --- A6 ghost vectors: soft-deleted multi-vectors remain physically present + attackable
exec_exp experiments.ghost_vectors          "ghost"            --n 40 --distractors 200 --font-size 34
# --- B6 certified bound: a provable privacy (epsilon) bound on the nullspace defense
exec_exp experiments.certified_bound        "certbound"        --ks 0,2,4,8,16,32 --n 40
# --- B8 real-doc defense transfer: nullspace P trained on synthetic, evaluated on real FUNSD
exec_exp experiments.defense_transfer_funsd "deftransfer"      --data "$FUNSD" --defense nullspace --n-train 64 --n-pages 40
# --- A7 transfer attack: attack a ColPali index with a proxy (different) encoder's queries
exec_exp experiments.transfer_attack        "transferatk"      --n 40 --distractors 200 --font-size 34
# cold alternative (single self-deleting VM per job): scripts/12_train_vm.sh us-central1-a a100 <job>

# --- multi-seed example for the PHASE-3 headline experiments (SEEDS defaults to "0 1 2 3 4") ----------
#   Uncomment to run the new headline experiments across seeds, then aggregate to mean +/- 95% CI as in
#   PHASE 1/2 above (Holm-Bonferroni over the claim family, COMPLETION_PLAN A2).
# for SEED in $SEEDS; do
#   exec_exp experiments.certified_defense "certified-seed${SEED}"   --ks 0,2,4,8,16,32 --n-train 64 --n-test 40 --seed "$SEED"
#   exec_exp experiments.transfer_attack   "transferatk-seed${SEED}" --n 40 --distractors 200 --font-size 34 --seed "$SEED"
# done
# exec_agg "agg-certified"   --runs "$RUNS" --metric-keys sweep.inverse_recovery,sweep.utility_topic_recall
# exec_agg "agg-transferatk" --runs "$RUNS" --metric-keys summary.name.top1_acc

# --- re-aggregate + re-draw figures so the paper figures include the PHASE-3 JSONs ------------------
echo "### local make_figures (refresh) — re-read every JSON incl. PHASE-3, rewrite paper/figures/*"
python3 -m experiments.make_figures --runs "$RUNS" --figdir paper/figures
echo "DONE (with PHASE 3). Figures in paper/figures/ ; per-experiment JSONs under ${RUNS}/."
