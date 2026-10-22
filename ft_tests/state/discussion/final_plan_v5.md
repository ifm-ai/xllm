# Artifact

Here is the corrected plan. Every verifier item is addressed below — the full self-contained v5 first, then the 34-row mapping table. Both of us proposed and checked every section; where we disagreed (the process-crash construction, the scaling ratio, the activation margin) the resolution is documented in the text. One new blocking issue surfaced during the rewrite and is included (§5.3).

---

# Final Plan: In-Band Fault Tolerance for xLLM (v5, self-contained)

*Supersedes v1–v4. Every number is **[measured]** (from `measured_all.md`, line cited), **[derived]** (inputs shown), or **[code]** (file:line). Torch library lines are from the installed torch **2.13.0+cu130** source in our reference path; the measured runs used torch **2.12** (`measured_all.md:1`) — library citations carry this caveat. `slurm.py` = `xllm/distributed/slurm.py`; `utils.py` = `xllm/utils.py` unless prefixed `distributed/`.*

---

## 0. What exists today (the baseline we must not break)

- **Restart-all is the only recovery.** Rank 0 runs `scontrol requeue` (`slurm.py:20-23`); SIGTERM is explicitly ignored (`slurm.py:28-30`); restarts are counted and resumed from the durable checkpoint (`utils.py:157-167` `get_n_restarts`, `checkpointing.py:371`). Matches the NVRx launcher: *"Only any-failed is supported"* (`refs/nvidia_resiliency_ext/fault_tolerance/launcher.py:1659-1660`).
- **NCCL-only.** All process groups are NCCL (`slurm.py:116-121`; `initialize.py:79,89,103,119`; `backend=None` passed at `train.py:113`, so the groups inherit the world's NCCL backend).
- **Static world.** Zero in-tree matches for elastic/spare/rendezvous/rejoin/dynamic world-size. `get_master_port`/`get_master_addr` are `@lru_cache`d (`slurm.py:87,97`); the master host is the first nodelist entry (`slurm.py:98-105`); the port is one draw from `random.Random(job_id)` (`slurm.py:88-94`, consumed at `:139`). **W is frozen at launch; a replacement must restore the same W.** The caches are **per-process**: a freshly started process recomputes rank, address and port (`slurm.py:47-97`), so they do not prevent a replacement node inside the allocation.
- **8 GPUs/node** is the config default `SlurmConf.gpus_per_node = 8` (`config.py:64`, also used by eval launching via the `async_eval_ngpus % gpus_per_node == 0` assert at `config.py:430`) and the hardware fact (`measured_all.md:13`). The `slurm.py:149` assert is `0 <= local_rank < 8` — an upper bound, not a fixed 8 (M0 removes it).
- **Durable checkpoint shard S_ckpt = 12 B/param** (fp32 param 4 + fp32 AdamW m,v 8): `utils.py:176` (fp32 default dtype), `optim/optimizer.py:5,26-32` (torch AdamW, no dtype override). **[measured]** 6.04 GB at 8B/16 ranks vs 6.00 GB predicted (0.7% error, `measured_all.md:7`). Distinct from the replica footprint S_rep = 12 B/param × 2 (own + peer copy) per rank — same units, different quantities (§5.3).

**The plan in one sentence:** keep a torn-safe replica of each rank's shard state (12 B/param GPU-resident + small training state) in node-local `/dev/shm`, written asynchronously every k steps by a per-node agent, so the common failures restart from local memory instead of requeueing the whole job — with today's restart-all preserved as the fallback at every level.

---

## 1. Measured inputs (the only measured numbers)

| quantity | value | source |
|---|---|---|
| Replica write cost, async, full (own+peer) | **2.3–2.6 s** of training time per copy at 8B/16r (S_rank = 6.04 GB) | [measured] `measured_all.md:7-8` |
| Same, synchronous | 3.0–3.3 s | [measured] `measured_all.md:7` |
| Async hiding | ~20–25% | [measured] `measured_all.md:8` |
| Own-slot copy only | 1.4–1.5 s per copy | [measured] `measured_all.md:8` |
| Marginal peer ship (C_peer) | 0.8–1.2 s from the async band; 1.5–1.9 s from the sync band | [derived] async full minus own-only; `measured_all.md:8` does not state whether own-only was sync or async |
| Step time (t_step) | **0.70 s** at 0.503B params/GPU (8B/16r) | [measured] `measured_all.md:7,14` |
| Node-death recovery, 8B on 3×8 (16 ranks; **spare node inside the allocation**) | **32.6 s**: detect 10.0, kill 10.54, PG init 5.41, build 0.58, restore 3.42, first step 2.68 — **process start excluded** | [measured] `measured_all.md:2-4` |
| Trainer process start (spawn + Lustre imports + CUDA init, 8/node) | 7.4–7.8 s | [measured] `measured_all.md:5` |
| PG scaling | re-init 0.4 s, xLLM groups 0.5 s, first all-reduce 0.65–0.87 s, destroy 1.1 s — **measured at 1–2 nodes only; 4–16 node points queued** | [measured] `measured_all.md:11-12` |
| Disk restore at 8B/16 GPUs | 47–53 s from Lustre DCP vs 3.4 s in-memory | [measured] `measured_all.md:9` |
| Node facts | /dev/shm 504 GB/node (GB/GiB as reported), 8× H100 80 GB/node, ~1 TB RAM | [measured] `measured_all.md:13` |
| GPU memory per rank, FSDP2, 8B/16r | 12.2 B/param between steps; 16.2 with grads; activation peak 47.8 GB (seq 2048, no AC) | [measured] `measured_all.md:14-16` |

All C_peer / marginal figures are **training-time costs per copy**, not transfer bandwidths. Everything at 1T/3T is extrapolated from the single 8B/16r point via `t_step ∝ P/W` and `C_total ∝ P/W` — two linearity assumptions with their settling test in §8.

---

## 2. Coverage matrix — 15 failure classes

T_rec values at 1T/512 are **[derived]** from the §6 decomposition (total 59.4 s). **No measured term dominates**: restore 22.3%, kill 17.7%, first step 17.6%, detect 16.8%, process start 12.6%, PG init 9.1%, build 3.8% (shares rounded, sum 99.9%). The risk sits in the three *unmeasured* terms (PG init at 512 nodes, NVMe bandwidth, activation peak — §8).

| # | failure | mechanism | recovery time | work lost | evidence | fallback |
|---|---|---|---|---|---|---|
| 1 | process crash | Own inactive `/dev/shm` slot, torn-safe protocol (§4); kill must be SIGKILL — SIGTERM ignored (`slurm.py:28-30`); survivors hang in NCCL and must be killed (`measured_all.md:6`) | **≤ 59.4 s** @1T/512 — same decomposition as row 3, except detection of a process exit may beat the deadline (unmeasured, §8) | ≤ k steps | design + code | restart-all (`slurm.py:20-23`) |
| 2 | GPU Xid/ECC | Node-health detection — **zero in-tree matches** for Xid/nvidia-smi/dmesg; surfaces as crash/hang | full T_rec | ≤ k | **design only**, named test (§8) | restart-all + requeue |
| 3 | node death | Peer replica restore, ring distance ≥ 2 (§4); **spare node inside the allocation** (the measured model, `measured_all.md:3`); replacement acquisition = process start 7.4–7.8 s (`measured_all.md:5`), included in T_rec; local-duplex rung: disk-spill read 37.5 s [5 GB/s node-aggregate, unmeasured §8] | **59.4 s** 3-slot; **83.7 s** local-duplex @1T/512 | ≤ k | derived (§6 table) | local-duplex → H0 |
| 4 | NCCL hang | Detection ≈ `nccl_timeout` = **3600 s default** (`config.py:376`, passed at `train.py:108,113`) → abort → crash lane | **~1 h + T_rec** at the default | ≤ k | code | restart-all |
| 5 | stuck-but-alive rank | Watchdog detects; kill must be SIGKILL (`slurm.py:28-30`) | T_rec | ≤ k | design only | process-crash lane |
| 6 | network partition | No partition detector; surfaces as NCCL timeout (3600 s default, `config.py:376`) | **~1 h + T_rec** | ≤ k | code + design | restart-all |
| 7 | filesystem outage | Replica lives in tmpfs ⇒ **training continues as long as nothing crashes**. A crash during an FS outage **cannot restart**: process spawn and imports come from Lustre (`measured_all.md:5`); data loading and the durable checkpoint also use the shared FS | 0 for FT while no crash | 0 while no crash | code | continue on replica; retry durable write |
| 8 | preemption / time limit | Today: SIGTERM ignored (`slurm.py:28-30`) → SIGKILL at grace end. **With M1.5 flush: lose ≤ 1 step** | grace window | ≤ k today; **≤ 1 with M1.5** | code | restart-all |
| 9 | two close failures | Survives two *adjacent* node deaths iff ring distance d ≥ 2; two deaths at exactly distance d lose primary + replica. **Assumption (stated, no evidence):** d ≥ 2 only moves the fatal pair from (i, i+1) to (i, i+d); for two independent failures the number of fatal pairs is still N — it helps only if failures correlate with index adjacency | T_rec or H0 | ≤ k or since last durable ckpt | derived from design | H0 |
| 10 | failure during a swap | Torn-safe protocol (§4): incomplete generation never adopted | 0 (run continues on prior generation) | one generation | design | previous generation |
| 11 | replacement node | **Spare-in-allocation (measured, `measured_all.md:3`) is the primary model** — the per-process `@lru_cache` (`slurm.py:87,97`) does not prevent it. **Requeue is the unmeasured fallback** (latency unmeasured, §8) | process start 7.4–7.8 s (in row 3); requeue unmeasured | ≤ k | measured + derived | restart-all |
| 12 | data-loader error | Loader state is in the replica (`train.py:167,191` `get_state`/`set_state` — §4) → resumable, not fatal | T_rec | ≤ k | code | restart-all |
| 13 | NaN / loss spike | **Only under `loss_rescaling=True`** (default False, `config.py:344`; with it off, `train.py:449` asserts scale == 1 and nothing is skipped): halve `state.scale`, skip step (`train.py:432-453`). **Existing bug:** the overflow `continue` (`train.py:453`) bypasses `optimizer.zero_grad()` (`train.py:468`), so overflowed grads accumulate into the next step — M0 fix. Fatal if `scale < 0.1` (`:438-442`) or non-finite loss (`:481`) | 0 in-loop; crash lane if fatal | skipped step(s) | code | bounded retries → H0 |
| 14 | agent crash | `/dev/shm` survives (tmpfs); respawned agent rescans slots and re-derives `resume_step` | seconds | 0 | design only; supervisor is an open item (§8) | node-level restart |
| 15 | store crash | Two-tier: **process crash, host alive** → respawn-in-place at the deterministic endpoint (`slurm.py:88-105`) + agents republish `resume_step` from slot meta; trainers never touch the store. **Host dead** → row 3 | ~0 (transparent) | 0 | design only | restart-all |

Notes:

- **Row 15's store is a TCPStore — a rendezvous service, not a process-group backend — so it does not violate NCCL-only.**
- The distinct FT store port requires a **second draw** (e.g. `random.Random(job_id+1)`); `get_master_port` returns exactly one port today (`slurm.py:88-94`). A small new code change, labelled as such.

---

## 3. The same check across parallelism, size, and memory regime

**Load-bearing precondition (parallelism invariance):** the replica is per-rank shard state, valid only if the shard layout is identical after restart. The layout is defined by `dp = W/(cp·tp)` (`config.py:383-387`), `replicated_size = dp/fss × cp` (`build.py:130`), the mesh construction (`build.py:126-132`, `distributed/utils.py:109-149`), and `W = fss × rs × tp` (`distributed/utils.py:115`) — all recomputed deterministically from the same config, **provided W is unchanged**, which the static-world design guarantees.

**The general fit quantity is params/GPU = P·cp / (fss·rs·tp)** (since W = fss·rs·tp, `distributed/utils.py:115`). It equals **P/W iff fss = dp and cp = 1** — a precondition, not the rule.

- **FSDP (fss = dp, cp = 1):** params/GPU = P/W; shards disjoint; replica = one shard per rank.
- **HSDP (fss < dp):** params/GPU = P/(fss·tp) — **larger** than P/W by dp/fss; the replica grows by the same factor, so the tmpfs limit of §5.3 tightens to **P/(fss·tp) ≤ 2.625e9**, not P/W. This is the case where the ceiling bites first.
- **CP (cp > 1):** params/GPU = P·cp/W (`build.py:130`, `distributed/utils.py:115`); at fixed W, CP **multiplies** parameter memory per GPU by cp (it folds into the HSDP replicate dim). CP splits activations only (`mappings.py:31,46,61,80,117,135`; `mha.py:118-120`). v4's "memory-neutral for params" is withdrawn.
- **TP (tp > 1):** params/GPU = P/(fss·rs·tp); replica per rank unchanged in form.
- **fss = 1:** legal only if cp = 1 (`build.py:128` assert), and it means DDP-like full replication across dp, not "TP-only."
- **Co-location:** the mesh is `arange(W).view(fss, rs, tp).swapdims(0,1)` (`distributed/utils.py:141-142`); a replica group spans rs·tp consecutive ranks, so single-node co-location requires 8 % (rs·tp) == 0 — "rs ≤ 8 ⇒ all replicas on one node" holds for tp = 1 only.
- **Invariance:** rows 1, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15 hold in every parallelism mode (12 of 15); rows 2, 4, 6 are detection-dependent; row 9 is the correlation-dependent one.

---

## 4. Replica contents and the torn-safe protocol

The replica is everything needed for a bitwise-faithful resume of a rank:

| state | source | why it must be in the replica |
|---|---|---|
| fp32 shard params | `train.py:160` `get_sharded_optimizer_state` | the weights |
| optimizer state: `exp_avg`, `exp_avg_sq`, **per-param `step`** | `torch/optim/adam.py:168-184` | `step` drives bias correction (`adam.py:771-780`); without it resume is not bitwise |
| LR scheduler state | `train.py:162`; mismatch **raises** `train.py:196-197` | a wrong scheduler step aborts the restart |
| `step`, `scale`, `scale_updates`, `clip_cumulative` | `train.py:163-166`, restored `:184-189` | loss-scale + clipping counters |
| **data-loader state** | `train.py:167` `get_state()`, restored `:191` `set_state()` | row 12's "resumable" claim depends on it |
| `timings` | `train.py:168`; optional on load (`:190` `.get`) | cheap, non-fatal |
| **MoE router bias** | `router.py:255` `register_buffer('bias', torch.zeros(num_experts))`, updated **in place** `router.py:195` (`router_bias.add_(bias_update, alpha=bias_update_rate)`) | training state that no optimizer state dict covers |
| other module buffers | FSDP2 has no `buffer_dtype`; buffers stay fp32 as created | e.g. norm/EMA buffers |

**Size consequence:** GPU-resident replica state is still **12 B/param** (params 4 + moments 8; the AdamW `step` is a CPU scalar, `adam.py:168-176`; scheduler/scaler/data/router-bias state is KB–MB per rank), so the §5.3 S_rank/S_node math is unchanged by the completed contents.

**Torn-safe protocol (inlined; no external reference):** each rank has slots A and B. A generation = the full replica payload + a meta block (generation counter, step, byte count, checksum). Write order: payload into the **inactive** slot, then meta **last**, atomically. A reader adopts a generation only if its meta is complete and self-consistent; otherwise it keeps the prior generation. A kill mid-write corrupts at most the incomplete generation.

---

## 5. Memory

### 5.1 GPU memory per rank — FSDP2 accounting

xLLM builds with FSDP2 (`fully_shard` + `MixedPrecisionPolicy`, `build.py:126-150`; no `offload_policy`, no flat params). The per-parameter bytes:

| component | B/param | lifetime | evidence |
|---|---|---|---|
| fp32 sharded parameter (DTensor, padded dim-0 chunk; **no flat params**) | 4 | persistent | built under `create_on_gpu()` (`build.py:144`) → default dtype fp32 (`xllm/utils.py:176`); `fully_shard` via `distributed/wrap.py:35`; shard in original dtype (`torch/.../_fsdp_param.py:298-328,588`) |
| AdamW `exp_avg`, `exp_avg_sq` (fp32) | 8 | persistent | `optim/optimizer.py:26-32` torch defaults; `zeros_like(p)` of the fp32 DTensor (`torch/optim/adam.py:178-184`) |
| fp32 sharded gradient | 4 | backward reduce-scatter → `optimizer.zero_grad()` (`train.py:468`) | view of `reduce_output`, cast to `orig_dtype` = fp32 (`_fsdp_collectives.py:696-754`) — fp32 because params are fp32, **not** because of `fp32_reduce_scatter` |
| foreach `exp_avg_sq_sqrt` temp | +4 | only inside `optimizer.step()` | `torch/optim/adam.py:790` (non-capturable path); foreach default via DTensor registration (`torch/distributed/tensor/__init__.py:74-75`) |
| bf16 param copy | 0 standing | per FSDP unit, freed on reshard | `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` `build.py:135-137`; cast at `_fsdp_param.py:896-901`; **no `_mp_shard` in FSDP2** |
| CPU offload | none | — | `fsdp_cfg` `build.py:138-142` has no `offload_policy` |
| **measured check, 8B/16r** | **12.2 / 16.2** | between steps / with grads | `measured_all.md:14-16`; the +4 foreach temp sits inside the step ⇒ 20 peak consistent |

**Answer: 12 B/param resident, 16 from backward through the step, 20 peak inside `optimizer.step`.** Footnotes: (i) the AdamW per-parameter `step` is a CPU scalar (`adam.py:168-176`) ⇒ in the replica for bias correction, **0 B/param** on GPU — do not double-count it; (ii) 1T/512 state = 1.953e9 × 20 = **39.1 GB**; (iii) 40 GB parts are excluded at every config; (iv) FSDP2's `fully_shard` has no `sync_module_states` — the v4 claim survives but is now cited at `build.py:126-132,138-142`.

**Activations — the 80 GB verdict is state-only and currently unproven.** Headroom at the 20 B/param state peak: 80 − 39.1 = **40.9 GB** [derived]. The only measured activation point is 39.7 GB at 8B/16r, seq 2048, no AC (47.8 − 8.1, `measured_all.md:16`), and it does **not** extrapolate by the plan's P/W rule — activations follow model shape × tokens/GPU × AC policy: params P ∝ L·d², stored activations per GPU ∝ L·d·tokens ⇒ **activations ∝ P/d**. 8B → 1T is 125× params: at fixed depth d grows 125^0.5 = 11.2× ⇒ activations ×11.2; if depth also doubles (d ×8), ×16. So **39.7 GB × 11–16 = 437–635 GB** at 1T/512, seq 2048, no AC [derived — model-shape scaling with an assumed architecture family], an ~11–16× shortfall against the 40.9 GB headroom. **Full activation checkpointing (reduction ≥ 11–16×) is unconditionally required at 1T/512, and the 80 GB fit is unproven until the peak is measured** (§8).

### 5.3 /dev/shm sizing — the replica, not the GPU, sets the minimum W

Per node the replica holds 8 own shards + 8 peer shards = 16 × 12 B/param × params/GPU = **192 B/param × params/GPU**, against /dev/shm = 504 GB/node (`measured_all.md:13`; GB/GiB as reported):

- Fit limit: params/GPU ≤ 504e9 / 192 = **2.625e9**.
- **1T/512:** params/GPU = 1.953e9 → S_node(own) = 12 × 1.953e9 × 8 = **187.5 GB**, S_node(own+peer) = **375 GB** = 74% of 504 GB ✓ (plus 256 MiB = 0.268 GB meta/slot buffer).
- **3T/512:** params/GPU = 5.859e9 → **1125 GB/node = 2.23× over** ✗. At W = 512 the design silently caps at P ≤ 2.625e9 × 512 = **1.34T**.

Since the requirement is no upper bound on model size, the plan states the condition explicitly:

- **W ≥ P / 2.625e9.** At 3T: W ≥ 1143 → **W = 2048** (params/GPU 1.465e9, 281 GB/node, 56% of tmpfs).
- **The replica, not GPU memory, sets the minimum W:** GPU state allows params/GPU ≤ 80/20 = 4.0e9; tmpfs allows 2.625e9 — the design needs **1.52× more GPUs** than memory alone would.
- **Cheaper default (recommended): store the peer copy only** — the own shard is recoverable from the peer that holds it. Per node = 96 B/param → limit 5.25e9 → **3T fits at W = 1024**, and 1T/512 drops to 187.5 GB (37%). Cost: own-slot restore becomes a peer fetch. Own+peer remains the belt-and-suspenders option.
- HSDP tightens the limit in the replicated dimension (§3): P/(fss·tp) ≤ 2.625e9.

---

## 6. Cadence and recovery-time decomposition

**Cadence.** Keep **k = 70**, derived from measured quantities only:

- k = C_total / (0.05 · t_step) = 2.45 / (0.05 × 0.70) = **70.0** — inputs: `measured_all.md:7` only. [derived]
- Copy overhead at k = 70: 8B 2.45/(70×0.70) = **5.00%**; 1T/512 9.57/(70×2.73) = **5.00%** — exactly scale-invariant because C ∝ P/W and t ∝ P/W. [derived, inputs shown]
- **Goodput = 1 − C_total/(k·t_step) − T_rec/MTBF_sys ≤ 95.0%** at every config. The MTBF term is **not computable**: no MTBF appears in `measured_all.md` (17 lines). 95.0% is a ceiling, not an estimate.
- Work lost per failure = k·t_step = 70 × 2.73 = **191 s (3.2 min)** at 1T/512.
- k = 8 overhead: 2.45/(8×0.70) = **43.75%** (t_step is **0.70** everywhere, never 0.699).
- **k\* is stated as a formula with MTBF a bare symbol — explicitly not a number in this plan.** v4's k\* = 1,337, the 78-node crossover, 0.023%, and the √W loss term are withdrawn: under Young/Daly the √ belongs to the *overhead at the optimal interval*; with a per-node MTBF, system MTBF = MTBF_node/N, so failures/time ∝ N and the recovery-loss term is **linear in N**.

**T_rec decomposition.** One scaling ratio, stated once: **3.9** = the 1T/512 to 8B/16r ratio the verifier already reproduced for t_step/C_total (`v4_verification.md:81`: 2.73/9.57 vs measured 0.70/2.45). Every P-scaled term uses it:

| term | 8B/3×8 [measured `measured_all.md:3`] | rule | 1T/512 |
|---|---|---|---|
| detect | 10.0 | deadline (`:3` "detect 6.0 = deadline"), not P-scaled | 10.00 |
| kill | 10.54 | **survivor termination** (survivors hang in NCCL, `measured_all.md:6`), not P-scaled | 10.54 |
| PG init | 5.41 | **unmeasured above 2 nodes** (`:11-12`); applied unscaled = floor | 5.41 |
| build | 0.58 | ∝ P/W ×3.9 | 2.26 |
| restore | 3.42 / 6.04 GB = **0.566 s/GB** | × 23.44 GB (S_rank at 1T/512) | 13.27 |
| first step | 2.68 | ∝ P/W ×3.9 | 10.45 |
| process start | 7.4–7.8 (`:5`) | excluded from the 32.6 s (`:2`) | 7.50 |
| **total** | 32.6 (+7.5) | | **59.4** |

- **Row 3 (node death) = 59.4 s** at 1T/512. Replacement-node acquisition is the process-start term (7.4–7.8 s, `measured_all.md:5`) on the spare inside the allocation — included.
- **Row 1 (process crash) = ≤ 59.4 s** — identical table (the kill term stays: survivors hang in 3 NCCL configurations, `measured_all.md:6`), own-slot restore at the same measured 0.566 s/GB (13.27 s); the only difference is that detection of a process exit may beat the 10.0 s deadline — unmeasured, §8.
- **Local-duplex = 59.4 − 13.27 + 37.5 = 83.7 s**, where 37.5 s = 187.5 GB / 5 GB/s node-aggregate NVMe (8 ranks share it; unmeasured assumption, §8).
- v4's 80.8 s, 104 s, 67.5 s, 48.9 s figures and the **41 s kill term are withdrawn** — measured kill is 10.54 s (`measured_all.md:3`), 17.7% of T_rec.
- Applied-unscaled terms (detect/kill/PG init) assume process/cluster operations independent of model size; PG init at 512 nodes is the term most likely to be an underestimate (§8).

---

## 7. Code changes

- **M0 (restart-all hardening):** (a) add the missing `_HYBRID_SHARD_DATA_PARALLEL_WORLD_SIZE` reset (`initialize.py:27`, cached `:218-219`) — `destroy_model_parallel` (`initialize.py:246-274`) already resets the MP/DP/CP groups, their world sizes and ranks, chunk ranks and the HSDP group; only that one variable goes stale; (b) remove the `xllm/distributed/slurm.py:149` assert (`0 <= local_rank < 8`, an upper bound); (c) fix the overflow path so skipped steps still zero grads (`train.py:453` `continue` bypasses `train.py:468` `optimizer.zero_grad()`).
- **M1 (skip-init):** apply to the value-filling inits at the four gaussian sites in `transformer.py`/`gekko.py` **plus** `causal_conv.py:72` and `residual/delta.py:94` (both `get_init_fn('gaussian', ...)` — a different call-site shape), `complex_exponential_moving_average.py:20-21,36` (raw `nn.init.normal_`), and `model_parallel/layers.py:138` (a default-argument `init.normal_` — a signature change, not a call-site change). **Design constraint:** `complex_exponential_moving_average.py:25` runs `dist.broadcast(idx_, src=0)` inside `_reset_parameters` — a collective that skip-init **must still execute**, or ranks diverge on the permutation.
- **M1.5 (preemption flush):** stop-step agreement — the stop step = the step in which the first SIGTERM arrives, published by that rank's agent to the store; survivors stop at the same published step. Grace need at 1T/512 = t_step + C_total = 2.73 + 9.57 ≈ **12.3 s**.

---

## 8. Open measurements (each blocks a named claim)

1. **GPU peak at 1T/512, seq 2048, with and without AC** — the 80 GB verdict is unproven until this lands (§5.1).
2. **tmpfs cgroup/epilog test** — nothing shows the job's memory cgroup allows 375–422 GB of /dev/shm, and Slurm epilogs often clean /dev/shm.
3. **NVMe node-aggregate bandwidth** (5 GB/s assumed in the local-duplex rung).
4. **PG init at 512 and 2048 nodes** (measured at 1–2 only, `measured_all.md:11-12`).
5. **Crash detection vs the 10.0 s deadline** — row 1 improves by up to the detect term if faster.
6. **tmpfs fit at 3T and the peer-only variant** (§5.3).
7. **Settling test for the two linearity assumptions** (t_step ∝ P/W, C_total ∝ P/W) at a second scale point.
8. **Unscaled-terms assumption** (detect/kill/PG init independent of model size) — spot-check at an intermediate size.
9. **Requeue latency** (row 11 fallback).
10. **Agent supervisor** (row 14).
11. **Xid/ECC detection** (row 2 — no in-tree mechanism exists today).
12. AdamW state dtypes citable at `torch/optim/adam.py:178-184`; the 0.566 s/GB restore rate sourced directly from `measured_all.md:3`.

---

## Mapping: all 34 verifier items → fixes

| # | v4 claim | verifier's correct fact (quoted) | fix in v5 |
|---|---|---|---|
| 1 | 16 B/param via flat params, `_mp_shard`, `buffer_dtype` | "Every citation is in `build_model_fsdp1` or torch FSDP1… 12 persistent, 16 after backward, 20 peak" | §5.1 rewritten: FSDP2 table, 12/16/20 with measured anchor `measured_all.md:14-16` |
| 2 | replica = params, moments, scale, clip, step | "Missing: the LR scheduler state… the data-loader state… the AdamW per-parameter `step`… and the MoE router-bias buffer" | §4 table completed, all code-cited |
| 3 | k=70 "5% overhead" + 99.0% goodput | "At k = 70 the copy term alone costs ~5%… so goodput is at most about 95%" | §6: goodput ≤ 95.0% ceiling, every input shown |
| 4 | 80.8 s / 104 s, "no spare injection possible" | "The measured 8B node-death run in fact used a spare node inside a 3×8 allocation" | spare-in-allocation primary; process start 7.5 s added; row 11 fixed |
| 5 | 80.8 s, 41 s kill, other T_rec ranges | "Show the decomposition… and the scaling rule used for each term" | §6 table; ratio 3.9 stated once; per-term rules |
| 6 | "41 s = 51% of T_rec" | "41/80.8 uses the 3-slot T_rec… 41/104 = 39%" | 41 s withdrawn; kill 10.54 s = 17.7% of 59.4 s |
| 7 | proc crash 67.5 s (80.8 − 13.3) | "Restore cost is treated as zero for the own lane… Add an own-slot restore term" | row 1 = ≤ 59.4 s with own-slot restore 13.27 s at measured 0.566 s/GB |
| 8 | `destroy_model_parallel` resets HSDP group only | "resets the MP, DP and CP groups, their world sizes and ranks… only `_HYBRID_SHARD_DATA_PARALLEL_WORLD_SIZE`" | M0 reversed: add that one reset (`initialize.py:27,218-219`) |

*(answer cut off at the 16384-token limit — raise Atlas's model.max_tokens)*

## Citation check

1 of 80 `file:line` references don't hold:
- `torch/.../_fsdp_param.py:298` — no such file in the workspace or reference paths
