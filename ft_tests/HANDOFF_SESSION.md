# Session handoff: everything the previous "Xllm" session was doing (for the session xllm_dev)

## The user and how to work
- Ahmad Chamma. Wants short, clear replies; step-by-step explanations when asked; every claim solid (measured, derived with
  inputs shown, or a cited code line); hesitations rediscussed until resolved; no unverified content in shared documents.
- Rules: never add AI/Claude attribution to commits, PRs or comments; one change per commit; do not push or open PRs without his
  OK; cancel only jobs you submitted, by id (never user-wide scancel); on m4 job names start with jt_, account jais, partition
  defq, --exclude=dgx-[61-62,101-106,108-109].

## Main project: xLLM training that never stops (hot swap)
Target: on a process/GPU/node failure, a per-node agent kills and restarts all trainers inside the same allocation (no requeue),
state comes back from host-memory copies (own /dev/shm + peer node), a dead node is replaced by an idle node requested on demand;
async disk checkpoints with bounded retention as safety net. Any model size (incl. 1T-3T, dense and MoE), any cluster size,
NCCL only (no gloo), generic (mostly NVIDIA), must not break xLLM (off by default, bitwise unchanged when off).
- Evidence and numbers: ft_tests/state/measured_all.md (first cluster). README.md lists the test scripts.
- Plan history: state/final_plan_v4.md (converged) -> independent verification found 34 problems (5 blocking):
  state/discussion/v4_verification.md -> a correction round is producing v5; it is auto-pushed to
  state/discussion/final_plan_v5.md when it ends. NEXT: re-verify v5 the same way (independent read-only agent checking every
  file:line against xllm main, torch sources, NVRx/torchft sources and measured_all.md); send remaining problems back to the
  models; repeat until clean.
- Key verified facts: in-process recovery after a real rank death FAILED (3 NCCL configs); restart-all recovery is bitwise and
  takes 6-33 s + 7-8 s process start (1B-8B); full copy at 8B costs 2.3-2.6 s async / 3.0-3.3 s sync per copy (own only 1.4-1.5 s);
  disk restore 47-53 s vs memory 3.4 s at 8B/16 GPUs; FSDP2 GPU memory 12 B/param resident, 16 during backward/step, 20 peak in
  optimizer.step; heartbeats must be counters (clock skew ~100 s); k* = sqrt(2*C*50700*3600/(W*t_step^2)) capped at the
  checkpoint interval.
- Open items: process-group scaling at 4/8/16+ nodes (first cluster is full; run on m4); 70B multi-node; xLLM's real data loader
  across a swap; two close failures; spare requested from the queue; agent and store crash paths; meta-device rebuild; chunked
  GPU packing buffers; replica contents completeness (LR scheduler, dataloader, AdamW step, MoE router bias, loss-scale state).

## Debate tool (Parley) and the two models
- Parley: github MBZUAI-Paris/Parley. Branches: ahmad.chamma/batch-operator-and-reference-paths (6 commits) and
  ahmad.chamma/operator-and-convergence-upgrades (12 more: NOT CONVERGED + extra-round offer, cross_review, Peer role,
  on_down wait, /attach, measured:/requirement: inbox labels, --task-file, status.json, parley continue, requirements checklist).
  Recommended config for these rounds: role Peer for both, governance.cross_review: true, round_limit: {offer: true},
  resilience.on_down: wait. Round template: state/discussion/parley_round_template.yaml; team instructions:
  state/discussion/PARLEY_team_instructions.md (put it as PARLEY.md in the workspace = a clean xllm main checkout).
- Models: Ahmad wants the discussion moved to m4: Kimi K3 on m4 (job 932555, jt_kimi_serve, belongs to the Jais-Type1 work but
  may be used) and the pre-existing DeepSeek on dgx-152 (check it is DeepSeek V4.1 Flash and note any difference). The same two
  models are still served on the first cluster (Kimi job 33020, DeepSeek job 32994) until a round works on m4; then ask Ahmad
  whether to stop them.
- Parley improvement ideas not done: assumption register (needs a design decision), auto-extend while objections shrink.

## Report (shareable claude.ai doc)
https://claude.ai/artifact/LjKhxkzkyg3bD7ondo7D4d (Claude Docs project 9fc666a8-2f02-4c72-9ecc-52433dbec112, tab file
5c586491-bc17, body node c8030f5b-f9c2). Contains only verified content: summary, status table, open decisions, today's gaps,
checkpoint benchmark chart, hot-swap GPU evidence table + recovery-timeline chart. Final target structure (Ahmad's request): full
migration plan readable by newcomers, architecture diagram, coverage of all cases, evidence from tests and discussion, scale
projections to 128/256/1024 nodes with error bands and plots, memory bounds, no-break contract, open items.

## Other threads (lower priority)
- PR #11 (Complete(d)P, ifm-ai/xllm) under review by Aditya; PR #17 (clip/base_lrs fixes). Replies were drafted earlier.
- xBridges commit dafa74b (fold residual multiplier at export) is local on the first cluster; needs write access.
- Confirmed review findings not yet fixed in xLLM: fp16 overflow skipping zero_grad (train.py:453), eval_freq=0 ZeroDivision.
- state/other/pr11_description_draft.md: updated PR #11 description draft (local only so far; Ahmad pastes it).
- state/other/xbridges_residual_fold.patch: the xBridges commit as a patch (apply with `git am` once there is write access).
