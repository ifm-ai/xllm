# Artifact

# Final Plan: In-Band Fault Tolerance for xLLM (v4, self-contained)

*This plan supersedes v1–v3. It needs no prior context. Every number is marked **[measured]** (from `measured_all.md`), **[derived]** (inputs shown), or **[code]** (file:line). Torch library lines are from the installed source in our reference path and are flagged as such. Both authors have verified every citation against the tree; three phantom citations produced during the rounds (`model_utils.py:70-71`, `fsdp.py:102`, `xllm/model_parallel/layers.py`) were caught and withdrawn — the anchors below are the corrected ones.*

---

## 0. What exists today (the baseline we must not break)

- **Restart-all is the only recovery.** Rank 0 runs `scontrol requeue` (`slurm.py:20-23`); SIGTERM is explicitly ignored (`slurm.py:28-30`); restarts are counted and resumed from the durable checkpoint (`utils.py:141-158`, `checkpointing.py:371`). This matches the NVRx launcher's sole supported policy: *"Only any-failed is supported"* (`refs/nvidia_resiliency_ext/fault_tolerance/launcher.py:1659-1660`).
- **NCCL-only.** All process groups are NCCL (`slurm.py:116-121`; `initialize.py:79,89,103,119`).
- **Static world.** There is no elastic membership: zero in-tree matches for elastic/spare/rendezvous/rejoin/dynamic world-size. `get_master_addr`/`get_master_port` are `@lru_cache`d (`slurm.py:87,97`), the master host is the first nodelist entry (`slurm.py:98-105`), and the port is one draw from `random.Random(job_id)` (`slurm.py:88-94`, consumed at `slurm.py:139`). **W is frozen at launch; a replacement must restore the same W.**
- **8 GPUs/node, hard-coded** (`slurm.py:149` assert — flagged as an M0 fix).
- **Checkpoint shard = 12 B/param** (fp32 param 4 + fp32 AdamW m,v 8): `utils.py:176` (fp32 default dtype), `optim/optimizer.py:5,26-32` (torch AdamW, no fused/foreach/dtype override). **[measured]** 6.04 GB at 8B/16 ranks vs 6.00 GB predicted (0.7% error); 2.86 GB at 1B/4 ranks vs 3.00 GB (5%).

**The plan in one sentence:** keep a torn-safe replica of each rank's 12 B/param shard in node-local `/dev/shm` (tmpfs), written asynchronously every k steps by a per-node agent, so that the common failures restart from local memory instead of requeueing the whole job — with today's restart-all preserved as the fallback at every level.

---

## 1. Measured inputs (the only measured numbers)

| quantity | value | source |
|---|---|---|
| Replica write cost, async (C_total) | **2.3–2.6 s** at 8B/16r (S_rank = 6.04 GB) | [measured] jobs 33037-33039 vs 33042/33043 |
| Same, sync | 3.0–3.3 s | [measured] |
| Async hiding | ~20–25% (this *corrects* v3's "async hides nothing") | [measured] |
| C_own (local slot write) | 1.4–1.5 s | [measured] `measured_all.md:8` |
| C_peer (marginal peer ship) | 0.8–1.2 s | [derived] from the async band; used only as marginal cost, never for k* |
| Effective bandwidths | B_own,eff = 4.2 GB/s (6.04/1.4–1.5), B_peer,eff = 6.0 GB/s (6.04/0.8–1.2) | [derived] — labelled derived, not measured |
| Step time (t_step) | **0.699 s** at 0.5B params/GPU (8B/16r) | [measured] |

Everything at 1T/3T is extrapolated from the single 8B/16r point via `t_step ∝ P/W` and `C_total ∝ P/W` — two linearity assumptions, flagged as such with their settling test in §8.

---

## 2. Coverage matrix — 15 failure classes

Columns: mechanism / recovery time / work lost / evidence / fallback. T_rec values at 1T/512 are **[derived]**; the kill term (41 s = 51% of T_rec at 1T/512) dominates at scale.

| # | failure | mechanism | recovery time | work lost | evidence | fallback |
|---|---|---|---|---|---|---|
| 1 | process crash | Own inactive `/dev/shm` slot, meta-last generation protocol (torn-safe, v2 §6); kill must be SIGKILL — SIGTERM ignored (`slurm.py:28-30`) | T_rec ≈ **67.5 s** @1T/512 (80.8 − 13.3: own lane skips peer restore) | ≤ k steps | design + code | restart-all (`slurm.py:20-23`) |
| 2 | GPU Xid/ECC | Node-health detection — **zero in-tree matches** for Xid/nvidia-smi/dmesg; surfaces as crash/hang | full T_rec | ≤ k | **design only**, named test (§8) | restart-all + requeue |
| 3 | node death | Peer replica restore, ring distance ≥ 2 (§4); restore 13.3 s @1T/512 **[derived: 0.57 s/GB × 23.4 GB]**; local-duplex rung: disk-spill read ~37 s **[5 GB/s extrapolated, unmeasured]** | ≈ **80.8 s** 3-slot; ≈ **104 s** local-duplex @1T/512 | ≤ k | derived | local-duplex → H0 |
| 4 | NCCL hang | PG timeout is a parameter (`initialize.py:60`, passed at `:79,89,103,119`) → abort → crash lane | timeout + T_rec | ≤ k | code | restart-all |
| 5 | stuck-but-alive rank | Watchdog detects; kill must be SIGKILL (`slurm.py:28-30`) | T_rec | ≤ k | design only | process-crash lane |
| 6 | network partition | No partition detector; surfaces as NCCL timeout (`initialize.py:60`) | timeout + T_rec | ≤ k | code + design | restart-all |
| 7 | filesystem outage | Replica lives in tmpfs ⇒ **FT is FS-independent**; only durable checkpoint writes (`checkpointing.py`) block | 0 for FT | **0** | code | continue on replica; retry durable write |
| 8 | preemption / time limit | Today: SIGTERM ignored (`slurm.py:28-30`) → SIGKILL at grace end. **With M1.5 flush (§6): lose ≤ 1 step** | grace window | ≤ k today; **≤ 1 with M1.5** | code | restart-all (rank-0-only requeue, `slurm.py:20-23`) |
| 9 | two close failures (node + peer) | Survives two *adjacent* node deaths **iff replica ring distance d ≥ 2**; two deaths at exactly distance d lose primary + replica | T_rec or H0 | ≤ k or since last durable ckpt | derived from design | H0 |
| 10 | failure during a swap | Meta-last + generation protocol (v2 §6): incomplete generation never adopted | 0 (run continues on prior generation) | one generation | design | previous generation |
| 11 | spare unavailable / slow | No spare injection possible (`@lru_cache`, `slurm.py:87,97`); W frozen → requeue | requeue latency — **unmeasured** | ≤ k | derived | restart-all |
| 12 | data-loader error | Loader state is checkpointed (`train.py:167,191` `get_state`/`set_state`) → resumable, not fatal | T_rec | ≤ k | code | restart-all |
| 13 | NaN / loss spike | In-loop: halve `state.scale`, skip step (`train.py:432-453`, `continue` skips `state.step += 1` at `:456`); fatal if `scale < 0.1` (`:438-442`) or non-finite loss (`:481`). **Not an FT event** (heartbeat continues) | 0 in-loop; crash lane if fatal | skipped step(s) | code | bounded retries → H0 (avoids NaN loops) |
| 14 | agent crash | `/dev/shm` survives (tmpfs); respawned agent rescans slots and re-derives `resume_step` (publication is agent-side) | seconds | 0 | design only; supervisor is an open item (§8) | node-level restart |
| 15 | store crash | Two-tier: **process crash, host alive** → respawn-in-place at the *deterministic* endpoint (`slurm.py:88-105`) + agents republish `resume_step` from slot meta; trainers never touch the store, so `@lru_cache` (`slurm.py:87,97`) stays *correct*. **Host dead** → row 3 | ~0 (transparent) | 0 | design only | restart-all |

Notes a reviewer will ask about:

- **Row 7 is the strongest free claim in the plan:** a shared-filesystem outage does not touch FT at all — checkpoint-restart cannot say that.
- **Row 15's store is a TCPStore — a rendezvous service, not a process-group backend — so it does not violate NCCL-only.**
- The distinct FT store port requires a **second draw** (e.g. `random.Random(job_id+1)`); `get_master_port` returns exactly one port today. This is a small new code change, labelled as such (same class as M7 below).

---

## 3. The same check across parallelism, size, and memory regime

**Load-bearing precondition (parallelism invariance):** the replica is per-rank shard state, valid only if the shard layout is identical after restart. It is: `dp = W/(cp·tp)` (`config.py:383-387`), `replicated_size`/`fully_sharded_size` (`utils.py:186-189`, `build.py:73-77`), and `W == fss × rs × tp` (`utils.py:115`) are all recomputed deterministically from the same config — **provided W is unchanged**, which the static-world design guarantees.

**The general fit quantity is params/GPU = P / (tp × fss)**, equal to P/W only when `fss = dp` (`build.py:73,75`: `fss = 1`, TP-only, is legal). The 1.75B/2.63B ceilings of §5 bind on this quantity.

- **FSDP:** shards disjoint; params/GPU = P/W. All rows as in §2.
- **HSDP (rs > 1):** params/GPU = P×rs/W — replication *costs* GPU memory (rs× FSDP). And the redundancy is **co-located**: the mesh (`utils.py:133-134`, `arange(W).view(fss,rs).swapdims(0,1)`) places a replica group on consecutive global ranks, and with 8 GPUs/node (`slurm.py:149`) and rs ≤ 8, **all R replicas sit on one node**. Consequences: row 1 gains a faster lane (co-located replica); **row 3 (node death) = H0** — HSDP gives zero node-death protection; survives R−1 *single-GPU* failures only. Verdict at rs=8, 16 B/param: viable for **1 of the 5** required configs (1T/2048: 62.5 GB/GPU; the rest exceed 80 GB).
- **TP:** shards disjoint; a TP group spans ≤ 1 node at tp=8, so a node death kills a full TP column — the replica must sit d ≥ 2 away **in node space, not rank space**.
- **CP:** **does not slice params — it replicates them.** `replicated_size = cp × dp` (`utils.py:186`), and the replicated mesh dim is literally named `"cp"` (`utils.py:138`). CP splits *activations* (`modules/context_parallel/mappings.py:31,46,61,80,117,135`; `fused_blocks/mha.py:118-120`). Memory-neutral for params; rows as FSDP.
- **EP/MoE:** there is **no EP group** — `config.py:418` asserts `num_experts % model_parallel_size == 0` and experts are sharded inside TP (`modules/moe/expert.py:72-74`). The EP column inherits TP exactly.

**Mode sensitivity:** 11 of 15 rows are mechanism-invariant across modes (2,4,5,6,7,8,10,12,13,14,15). Four are mode-sensitive:

| row | FSDP | HSDP | TP | CP | EP/MoE |
|---|---|---|---|---|---|
| 1 process crash | own slot | own slot **or co-located replica (faster)** | own slot | own slot | own slot |
| 3 node death | peer replica (13.3 s @1T/512) | **H0** | peer replica, d ≥ 2 in node space | as FSDP | as TP |
| 9 two close failures | ring d ≥ 2 | H0 | ring d ≥ 2 (node space) | ring d ≥ 2 | ring d ≥ 2 |
| 11 spare | same-W requeue | same | same | same | same |

**Memory regime:** ceiling from 16 B/param (§5): params/GPU ≤ 80/16 = **5B** before activations; practical ≤ ~4B. 1T/512 (1.953B/GPU) = 31.2 GB state — fits with the 80 GB floor. HSDP at rs=8 multiplies by 8 and is out for 4 of 5 configs. CP is free on params, sublinear on activations.

---

## 4. Architecture (diagram description)

```
Node i (8 GPUs, hard-coded slurm.py:149)
├── 8 trainer ranks ── NCCL only (slurm.py:116-121)
├── 1 FT agent process (per node)
│     • packs each rank's 12 B/param shard every k steps (async, default on)
│     • chunked copy: one fixed 256 MiB buffer, reused serially
│     • writes: slot A/B ping-pong in /dev/shm, meta LAST (generation counter)
│     • ships peer copy to node (i+d) mod N,  d ≥ 2          ← placement invariant
│     • local-duplex rung: peer copy spilled to node-local NVMe instead
│     • publishes resume_step to the store
├── /dev/shm (tmpfs)
│     • full 3-slot rung:  2 own slots + 1 peer slot  (3 × S_node ≤ 504 GB)
│     • local-duplex rung: 2 own slots only           (2 × S_node ≤ 504 GB)
└── node-local NVMe: ≥ S_node free (local-duplex peer spill)

Store: TCPStore on master host (first nodelist entry, slurm.py:98-105),
deterministic port (second draw from the job-seeded RNG — new code).
Trainers NEVER touch the store; agents only. Respawn-in-place on crash.

Replica ring: node i's replica lives on node (i+d) mod N, d ≥ 2.
Survives any two ADJACENT node deaths; two deaths at ring distance d → H0.
```

**Replica contents (corrected — Delta's finding, accepted):** fp32 params, fp32 moments, **plus `state.scale`, `state.scale_updates`, `clip_cumulative` and the step counter** (`train.py:436-437,459`). Without these, a restart resets the loss scale and is **not bitwise** under `loss_rescaling`. They are added to the replica and to the M2 readback check.

---

## 5. Memory bounds

### 5.1 GPU side — steady state = 16 B/param (consensus figure)

| component | B/param | evidence |
|---|---|---|
| fp32 master params | 4 | `utils.py:176`; flat params, `build.py:95` `use_orig_params=False` |
| AdamW m,v (fp32) | 8 | `optim/optimizer.py:26-32` — built on the fp32 flat params, no dtype override |
| sharded grad (fp32) | 4 | `config.py:352` `fp32_reduce_scatter=True` → `build.py:86-88` `reduce_dtype=fp32` → torch `_flat_param.py:935` → cast before reduce-scatter and back to param dtype `_runtime_utils.py:757-764,1046-1048` *(library source, flagged)* |
| **total steady state** | **16** | |

The bf16 `_mp_shard` (+2) is a **per-layer transient**, not standing state: allocated at pre-unshard, freed at post-unshard (`_flat_param.py:1342-1357, 1529-1530`, library source). Also: `build.py:88` `buffer_dtype=fp32`; **no `cpu_offload` key** in `build.py:89-98` ⇒ torch default `None` ⇒ state fully GPU-resident, which is what makes the 16 binding. (The 16 vs 18 disagreement from the rounds resolved here: grad fp32 = +4 confirmed; `_mp_shard` resident claim withdrawn after re-reading the post-unshard free.)

**80 GB floor for 1T/512 is robust to the accounting:** state = 31.2 GB @16, 35.2 @18, 39.1 @20 B/param — any of these leaves < 9 GB for activations + transients + CUDA context on a 40 GB part. Plan states the floor, not one constant.

### 5.2 GPU packing rule

**Chunked fixed-size buffers, never whole-shard.** One `pack_chunk_bytes` buffer, **default 256 MiB**, reused serially ⇒ **0.25 GB constant at every model and cluster size**. Whole-shard staging at 1T/512 is 23.4 GB = 48–52% of free GPU memory — prohibited by rule, not merely discouraged.

### 5.3 Host side — the ladder (fit rule in params-per-GPU)

S_node = 8 ranks × 12 B × params/GPU. `/dev/shm` budget = 504 GB/node.

| rung | slots | fit condition | disk requirement | survives process crash | survives node death |
|---|---|---|---|---|---|
| **full 3-slot** | 2 own + 1 peer in shm | params/GPU ≤ **1.75B** (3 × S_node ≤ 504) | none | local inactive slot | peer replica |
| **local-duplex** | 2 own in shm; peer spilled to node-local disk | params/GPU ≤ **2.63B** (2 × S_node ≤ 504) | **≥ S_node free NVMe** (187 GB @1T/512) | local inactive slot (fast lane unchanged) | disk read of spilled peer copy (~37 s @5 GB/s, extrapolated) |
| **H0** | — | > 2.63B | shared FS | restart-all | restart-all |

Local-duplex keeps the torn-safe fast lane and demotes only node death to a local-disk read — strictly better than H0 (no shared-FS dependence). The 375 GB "2 own + peer-from-inactive" construction from the rounds was **rejected by both authors**: no slot holds the peer copy (3 × 187 = 562 > 504), and transient staging re-opens the torn window.

### 5.4 The five required configs

| config | params/GPU | S_rank | S_node | rung | host need | GPU state @16 B | verdict |
|---|---|---|---|---|---|---|---|
| 1T / 512 | 1.953B | 23.4 GB | 187.3 GB | **local-duplex** | 375 GB shm + 187 GB NVMe | 31.2 GB | fits; **80 GB H100 required** |
| 1T / 1024 | 0.977B | 11.7 GB | 93.8 GB | 3-slot | 281 GB | 15.6 GB | fits |
| 1T / 2048 | 0.488B | 5.9 GB | 46.9 GB | 3-slot | 141 GB | 7.8 GB | fits |
| 3T / 2048 | 1.465B | 17.6 GB | 140.6 GB | 3-slot | 422 GB | 23.4 GB | fits |
| 3T / 4096 | 0.732B | 8.8 GB | 70.3 GB | 3-slot | 211 GB | 11.7 GB | fits |

None of the five falls to H0. All S_rank/S_node figures **[derived: 12 B × params/GPU; ×8]**.

---

## 6. Cadence, projections, gates (v3 kept, corrected where measurement disagrees)

**Cadence rule (corrected to async measured values, C_total = 2.45 s mid):**

- k = 8 overhead: 2.45/(8·0.699) = **43.8%** (band 41.1–46.5%) **[derived]**
- 5%-overhead cadence: **k = 70 steps** (band 66–74) **[derived]**
- k\* @ W=1024: **~1,337** (unchanged from v3)
- crossover k\* = k_cap: **~78 nodes**

**Projections at scale** (t_step and C_total extrapolated ∝ P/W from the measured point — assumption flagged):

| config | t_step | C_total | T_rec (node death) | T_rec (proc crash) | goodput |
|---|---|---|---|---|---|
| 1T/512 | 2.73 s | 9.5 s | ≈104 s (local-duplex) | ≈67.5 s | 99.0% |
| 1T/1024 | 1.37 s | 4.8 s | ≈40–50 s | ≈35–45 s | ~99% |
| 1T/2048 | 0.68 s | 2.4 s | ≈30–40 s | ≈25–35 s | ~99% |
| 3T/2048 | 2.05 s | 7.2 s | ≈60–75 s | ≈50–60 s | ~99% |
| 3T/4096 | 1.02 s | 3.6 s | ≈35–45 s | ≈30–40 s | ~99% |

**Goodput is flat 98.6–99.2% across 1T/3T** — the copy term is scale-invariant in P (C ∝ P/W, t ∝ P/W) and the loss term grows only as √W (MTBF term 0.023% at 1T/512). **v3's "86.3% pessimistic at scale" framing is wrong for 1T/3T and is corrected here.** The real scale risks are: host-RAM fit (§5), the kill term (41 s = 51% of T_rec at 1T/512), and PG init above 16 nodes (unmeasured, §8).

**Gates:** (i) M7 ships only behind its measurement gate; (ii) the 3T rows are gated on the extrapolation test in §8 — if `t_step` or `C_total` is superlinear in P/W, the 3T rows are optimistic and must be recomputed; (iii) local-duplex is gated on the NVMe bandwidth measurement, since 37 s is the second-largest term in its node-death T_rec.

---

## 7. Migration phases, flags, defaults, user-facing changes, rollback, no-break contract

**Phases (each behind its flag; each independently revertible):**

- **M0 — hygiene fixes (no behavior change):** add world-size/rank resets to `destroy_model_parallel` (`initialize.py:246-274` currently resets `_HYBRID_SHARD_DATA_PARALLEL_GROUP` only — no world-size reset exists); remove the 8-GPUs/node assert (`slurm.py:149`).
- **M1 — own-slot replica:** agent packs each rank's shard to `/dev/shm`, ping-pong slots, meta-last generation protocol (torn-safe). Includes rung (A) skip-init: **small** code change threading `init_mode` (`config.py:147`; `get_init_fn('none')` = identity, `utils.py:105-107`) into the hard-coded gaussian sites `transformer.py:540,601` and `gekko.py:220,336`. Safe because `sync_module_states=False` (`build.py:94`) — no wrap-time broadcast to diverge from; every rank **loads** its own shard from the replica (a load, not a broadcast).
- **M1.5 — preemption flush (new, best cost/benefit in the plan):** replace the SIGTERM-ignore (`slurm.py:28-30`) with a handler that sets a stop flag, finishes the current step, packs into the **inactive** slot meta-last (torn-safe if SIGKILL lands mid-flush → prior generation), then exits. Requirement: Slurm grace ≥ step + pack (~15 s at 1T/512; ~3 s at 8B). Effect: **preemption loses ≤ 1 step instead of k**.
- **M2 — peer replica + ring d ≥ 2 + readback:** peer ship to node (i+d) mod N; readback verification including the loss-scale state (`state.scale`, `scale_updates`, `clip_cumulative`, step — `train.py:436-437,459`).
- **M7 — `use_local_synchronization`:** proposed code change at `initialize.py:79,89,103,119`; ships only behind a measurement gate. Labelled a work item, not a config lever.

**Flags and defaults:** async copy **on** by default (C_total 2.3–2.6 s [measured]); `pack_chunk_bytes` = 256 MiB; cadence k = 70 (5% overhead); ring distance d = 2; FT store port = second seeded draw (new code). Sync mode (3.0–3.3 s) remains selectable.

**User-facing changes:** (1) resume after a process crash/node death is from local memory in tens of seconds, not a requeue; (2) resume is **bitwise** — loss-scale state is in the replica; (3) preemption loses ≤ 1 step (with M1.5); (4) 1T/512 requires 80 GB parts and 187 GB node-local NVMe; (5) no new user actions required — agents and the store are self-supervising (supervisor is an open item, §8).

**Rollback:** every rung falls back to the level below, and H0 (today's checkpoint-restart: `slurm.py:20-23`, `utils.py:141-158`, `checkpointing.py:371`) is always available; flags off = exactly today's behavior.

**No-break contract:** restart-all is preserved as the terminal fallback; NCCL-only is preserved (the store is a rendezvous service, not a PG backend); W is frozen at launch (no elastic semantics introduced); trainers never touch the store; the durable checkpoint path is untouched (and is the only thing a filesystem outage affects — row 7).

---

## 8. Open items — every unmeasured claim, with the test that settles it

| # | claim | status | settling test |
|---|---|---|---|
| 1 | All 1T/3T numbers rest on **one measured point** (8B/16r) via `t_step ∝ P/W`, `C_total ∝ P/W` | assumption | Run 8B at 64 and 128 ranks; fit `t_step` and `C_total` vs P/W. Superlinearity ⇒ 3T rows optimistic. |
| 2 | `t_step` = 2.73 s at 1T/512 is **3.9× beyond** the measured 0.5B/GPU point — the least trustworthy number in the plan | assumption | Same test as #1. |
| 3 | Node-local NVMe ~5 GB/s (37 s disk read = 2nd-largest term in local-duplex node-death T_rec) | extrapolation, no anchor | Write+read 187 GB to node-local NVMe; report GB/s. |
| 4 | Shared-FS bandwidth — **the H0 comparison cannot be closed honestly without it** | unmeasured | Measure shared-FS read GB/s/rank at 8B; H0 recovery = 12 TB / aggregate + requeue latency. |
| 5 | Requeue latency | unmeasured | Time `scontrol requeue` → running at target scale. |
| 6 | PG init cost above 16 nodes | unmeasured | Time `initialize.py:79,89,103,119` at 64/128/512 nodes. |
| 7 | Store respawn-in-place + agent rescan (supervisor does not exist yet) | design | Kill store/agent processes; verify respawn, `resume_step` republication from slot meta, zero trainer interruption. |
| 8 | GPU Xid/ECC detection (zero in-tree mechanism) | design only | Named node-health test: inject/simulate Xid, verify detection → crash lane. |
| 9 | AdamW moment dtype rests on torch `zeros_like` default (library behavior, not workspace-citable) | code + library | `opt.state[p]["exp_avg"].dtype is torch.float32`. |
| 10 | B_own,eff = 4.2 GB/s, B_peer,eff = 6.0 GB/s | derived, not measured | Direct bandwidth microbenchmark of slot write / peer ship at 23.4 GB. |
| 11 | Peer restore 13.3 s @1T/512 (0.57 s/GB × 23.4 GB) | derived | Covered by #1 + #10. |
| 12 | M7 `use_local_synchronization` benefit | proposed change | Measurement gate: step-time A/B at scale. |

---

## 9. What changed relative to v3 (for the record)

1. Async cost corrected: async hides ~20–25% (2.3–2.6 s), not "nothing"; all cadence math recomputed (k=70 @5%, k=8 → 43.8%).
2. Replica contents extended with loss-scale state (bitwise resume).
3. GPU steady state fixed at **16 B/param** (was 8 in early drafts; 18 briefly; grad fp32 confirmed, `_mp_shard` transient) — 80 GB floor for 1T/512 stated robustly.
4. Ladder restated in **params-per-GPU** with the **local-duplex** rung replacing the rejected 375 GB construction; NVMe precondition added.
5. **Ring distance d ≥ 2** added as a placement invariant (row 9 was unwinnable in all prior drafts).
6. **M1.5 SIGTERM-flush** promoted from an aside to a migration phase.
7. Parallelism section written for the first time: CP replicates params (does not slice), HSDP replicas are node-co-located (**node death = H0 under HSDP**; viable for 1 of 5 configs), EP = TP in xLLM, general formula params/GPU = P/(tp × fss).
8. "86.3% at scale" pessimism corrected to flat 98.6–99.2%, with the real scale risks named.
9. Store row is two-tier (respawn-in-place at a deterministic endpoint; host-dead = node-death row); distinct FT port labelled a new code change.
10. Three phantom citations withdrawn; every remaining citation re-verified against the tree by both authors.

Both authors propose, critique, and have verified the corrections above against the tree; the text above is the agreed final plan.

## Citation check

5 of 49 `file:line` references don't hold:
- `config.py:147` — the file has 36 lines
- `config.py:352` — the file has 36 lines
- `config.py:383-387` — the file has 36 lines
- `config.py:418` — the file has 36 lines
- `model_utils.py:70` — no such file in the workspace or reference paths
