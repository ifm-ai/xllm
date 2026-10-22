# Verification of "Final Plan: In-Band Fault Tolerance for xLLM (v4)"

Plan checked: `teamruns_faulttol8/parley_20261022-185322_final-completeness-round-inputs-final-pl/artifact.md`
Sources: xLLM `xllm-review` @ f5f8687 (paths below are relative to that repo unless prefixed `torch/`); torch = installed `hht` env (`torch/version.py`: **2.13.0+cu130**; measured_all.md says runs used torch 2.12); NVRx under `ft_round7/refs/`; numbers from `ft_round7/measured_all.md` only.

Note on ambiguous paths: the plan writes `utils.py` for two different files. `utils.py:105-107, 141-158, 176, 186-189` are `xllm/utils.py`; `utils.py:115, 133-134, 138` are `xllm/distributed/utils.py`.

## Summary

- 48 `file:line` citations checked (counting each cited location once).
  - 37 correct (3 of these are torch FSDP1 lines: correct text, but they do not apply to xLLM's default FSDP2 path).
  - 5 wrong line, but the claim is true (correct line given below).
  - 6 claim false or overstated.
- The plan's own "Citation check" footer is wrong. `xllm/config.py` has 444 lines, and all four `config.py` citations (147, 352, 383-387, 418) are correct. `model_utils.py:70` appears only in the note withdrawing it.
- Problems found: 5 blocking, 18 should-fix, 11 minor (table below).

## FSDP2 GPU memory: what is on each GPU per parameter (default `ddp_backend='fsdp2'`, `config.py:108`)

| state | bytes/param | lifetime | evidence |
|---|---|---|---|
| fp32 sharded parameter (DTensor, padded dim-0 chunk; no flat params) | 4 | persistent | Model is built under `create_on_gpu()` (`models/build.py:144`), which sets the default dtype to fp32 (`xllm/utils.py:176`). `fully_shard` is applied by `distributed/wrap.py:35` to embed, each layer, output and root (`models/transformer.py:602,619,621`; `models/build.py:147`). The shard is created at `torch/distributed/fsdp/_fully_shard/_fsdp_param.py:298-328` in its original dtype (`_fsdp_param.py:588`). |
| AdamW `exp_avg`, `exp_avg_sq` (fp32) | 8 | persistent | `xllm/optim/optimizer.py:26-32` uses torch AdamW defaults. The moments are `zeros_like(p)` of the fp32 DTensor (`torch/optim/adam.py:178-184`). The per-parameter `step` is a CPU scalar (`adam.py:168-176`). |
| fp32 sharded gradient | 4 | From the backward reduce-scatter until `optimizer.zero_grad()` (`train.py:468`, set_to_none default), so it is freed every step. | The grad is a view of `reduce_output`, cast to `orig_dtype` = fp32 (`torch/.../_fully_shard/_fsdp_collectives.py:696-754`). It is fp32 because the parameters are fp32, **not** because `fp32_reduce_scatter=True`: with bf16 reduction it is still cast back to fp32 at `:696`. |
| foreach AdamW temporary `exp_avg_sq_sqrt` | +4 | Only inside `optimizer.step()` | foreach is the default because DTensor is registered as foreach-capable (`torch/distributed/tensor/__init__.py:74-75`). The temporary is allocated at `torch/optim/adam.py:790` (non-capturable path; `:762` capturable). |
| bf16 parameter copy | 0 standing | Transient per FSDP unit: cast as the all-gather input on unshard, freed on reshard (`reshard_after_forward=True`, `config.py:353`) | `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` at `models/build.py:135-137`. The cast is at `_fsdp_param.py:896-901`. FSDP2 has no `_mp_shard`. |
| fp32 reduce-scatter input | 0 standing | Transient per FSDP unit in backward | `_fsdp_collectives.py:597-606` |
| CPU offload | none | n/a | `fsdp_cfg` (`models/build.py:138-142`) has no `offload_policy`, so the default `OffloadPolicy()` applies, which means no offload (`torch/.../_fully_shard/_fully_shard.py:105`; `_fsdp_api.py:175-179`). |
| buffers | negligible | persistent, replicated | FSDP2 has no `buffer_dtype`. Buffers stay fp32 as created. |

**Answer.**
- 12 B/param stays on the GPU for the whole iteration: the fp32 parameter plus the two AdamW moments.
- From the backward pass through `optimizer.step()` there are 16 B/param, because the fp32 gradient is added.
- The peak is 20 B/param, inside `optimizer.step()`, because of the foreach `sqrt` temporary.
- So "16 B/param" holds numerically as the post-backward resident figure. It is not standing state, and it is not the peak.
- Every mechanism the plan cites for it is FSDP1-only: flat params, `use_orig_params`, `MixedPrecision.buffer_dtype`, `_flat_param.py`, `_runtime_utils.py`, and `build.py:85-98` in `build_model_fsdp1`.
- The plan's conclusion that 1T/512 needs an 80 GB GPU still holds: 39.1 GB at 20 B/param.

## Problems

| # | § | claim in plan | what is wrong | correct fact (citation) | severity |
|---|---|---|---|---|---|
| 1 | 5.1 | The GPU state is 16 B/param via flat params (`build.py:95`), the `reduce_dtype` chain (`build.py:86-88`, `_flat_param.py:935`, `_runtime_utils.py:757-764,1046-1048`), `_mp_shard` (`_flat_param.py:1342-1357,1529-1530`), `buffer_dtype` (`build.py:88`) and no `cpu_offload` (`build.py:89-98`). | Every citation is in `build_model_fsdp1` or torch FSDP1. The default backend, and every measurement, is FSDP2 (`config.py:108`; `build_model_fsdp2` at `build.py:109-150`). FSDP2 has no flat params, no `_mp_shard` and no `buffer_dtype`. Its fp32 gradient comes from `orig_dtype`, not from `fp32_reduce_scatter`. | See the FSDP2 section above: 12 persistent, 16 after backward, 20 peak in `optimizer.step`. | blocking |
| 2 | 4, 7 | The replica holds params, moments, `scale`, `scale_updates`, `clip_cumulative` and `step`, so resume is "bitwise". | This is incomplete. Missing: the LR scheduler state (`train.py:162`; a mismatch raises at `train.py:196-197`), the data-loader state (`train.py:167`, which row 12 relies on), the AdamW per-parameter `step` used for bias correction (`torch/optim/adam.py:168-176`), and the MoE router-bias buffer, which is updated in training (`modules/moe/router.py:195,255`). | At minimum the replica must contain everything in `training_state_dict` (`train.py:161-169`), the full optimizer state including `step`, and the module buffers. | blocking |
| 3 | 6, 7 | Default cadence k = 70 ("5% overhead") and goodput of 99.0% at 1T/512 ("98.6–99.2%" across configs). | These contradict each other. At k = 70 the copy term alone costs ~5% (2.45/(70·0.699) = 5.0%), so goodput is at most about 95%. The 99% figure seems to assume k ≈ k\* (~1,337), but no input for that is given. | Either the default is k\* and the cadence rule is restated, or the goodput is recomputed at k = 70. | blocking |
| 4 | 2 (rows 3, 11), 0 | Node death recovers in 80.8 s (3-slot) or 104 s (local-duplex) at 1T/512, while "no spare injection possible; W frozen → requeue". | The node-death T_rec leaves out getting a replacement node. If, as row 11 says, the only way is a requeue, then the requeue latency (unmeasured, §8 #5) and process start (7.4-7.8 s, `measured_all.md:5`) belong in row 3. The measured 8B node-death run in fact used a spare node inside a 3×8 allocation for 16 ranks (`measured_all.md:3`). So spare nodes within the allocation are possible, and `@lru_cache` (which is per-process) does not prevent them. | Pick one model: a spare node in the allocation (measured) or a requeue (unmeasured). Then state T_rec with that term included. | blocking |
| 5 | 2, 6 | The 1T/512 T_rec of 80.8 s, the "kill term 41 s = 51% of T_rec", and the T_rec ranges for the other 4 configs (≈40–50, 30–40, 60–75, 35–45 s). | No inputs are given for 80.8 s, for the 41 s kill term, or for any of the other T_rec ranges. The measured kill time is 10.54 s at 8B on 3×8 GPUs (`measured_all.md:3`). The kill term is the largest term, but it does not appear in the §8 open-items table. | Show the decomposition (detect, kill, PG init, build, restore, first step, all from `measured_all.md:3`) and the scaling rule used for each term. | blocking |
| 6 | 2 header | "41 s = 51% of T_rec at 1T/512" | 41/80.8 uses the 3-slot T_rec. §5.4 puts 1T/512 on the local-duplex rung, where node-death T_rec is 104 s, so the share is 41/104 = 39%. | | should-fix |
| 7 | 2 row 1 | Process crash T_rec = 67.5 s (80.8 − 13.3, because the own lane skips the peer restore) | Restore cost is treated as zero for the own lane. Restoring 23.4 GB/rank from local `/dev/shm` still has to be added. | Add an own-slot restore term. The only measured restore is 3.42 s for 6.04 GB/rank (`measured_all.md:3`). | should-fix |
| 8 | 7 M0 | `destroy_model_parallel` (`initialize.py:246-274`) "resets `_HYBRID_SHARD_DATA_PARALLEL_GROUP` only — no world-size reset exists" | False. | The function resets the MP, DP and CP groups, their world sizes and ranks, the prev/next chunk ranks and the HSDP group (`initialize.py:248-274`). The only state it does not reset is `_HYBRID_SHARD_DATA_PARALLEL_WORLD_SIZE` (`initialize.py:27`, cached at `:218-219`). Separately, `slurm.py` caches rank and world size with `@lru_cache` (`slurm.py:52-79`). | should-fix |
| 9 | 3 | params/GPU = P/(tp×fss) "equal to P/W only when fss = dp"; "CP: memory-neutral for params"; "CP is free on params" | In FSDP2, CP is folded into the HSDP replicate dimension: `replicated_size = dp/fss × cp` (`build.py:130`), and W = fss·rs·tp (`distributed/utils.py:115`). So params/GPU = P·cp/W when fss = dp. At a fixed W, CP multiplies the parameter memory per GPU by cp. | P/W holds only when fss = dp **and** cp = 1. | should-fix |
| 10 | 2 row 13 | NaN or loss spike: halve the scale and skip the step in-loop, so it is "not an FT event". | This only applies with `loss_rescaling=True`. The default is False (`config.py:344`), and in that case `train.py:449` asserts scale == 1 and does not skip anything. Also, the overflow `continue` (`train.py:453`) bypasses `optimizer.zero_grad()` (`train.py:468`), so the overflowed gradients are accumulated into the next step (`_fsdp_collectives.py:744-749`). The step is not cleanly skipped. | State the condition, and note the existing missing `zero_grad` on overflow. | should-fix |
| 11 | 7 M1 | Skip-init only needs `init_mode` threaded into `transformer.py:540,601` and `gekko.py:220,336`. | Those four lines are correct, but the list is incomplete. Other init sites ignore `init_mode`. | Also: `modules/causal_conv.py:72`, `modules/residual/delta.py:94`, `modules/complex_exponential_moving_average.py:20-21,36`, and the default `init.normal_` at `modules/model_parallel/layers.py:138`. | should-fix |
| 12 | 3, 7 M1 | `build.py:73-77`, `build.py:73,75`, `build.py:94` (`sync_module_states=False`) | These are FSDP1 lines. | FSDP2 equivalents: `build.py:126-130`. FSDP2's `fully_shard` has no `sync_module_states` and does no broadcast when wrapping (`fsdp_cfg` at `build.py:138-142`). The claim holds; the citation is wrong. | should-fix |
| 13 | 3 | `replicated_size`/`fully_sharded_size` "recomputed deterministically" (`utils.py:186-189`); "CP replicates: `replicated_size = cp × dp` (`utils.py:186`)" | `xllm/utils.py:183-192` is `num_parameters()`, which only counts parameters. Under fsdp2 it does not divide by fss. It does not define the layout. | The layout is defined at `models/build.py:126-132` and `distributed/utils.py:109-149`. | should-fix |
| 14 | 6 | k\* at W=1024 is ~1,337; the crossover is ~78 nodes; the "MTBF term" is 0.023%; "the loss term grows only as √W"; goodput is 98.6–99.2%. | No inputs are given (no MTBF, no k_cap, no formula), so none of these can be reproduced. The √W scaling is asserted without derivation; under Young/Daly the recovery-loss term T_rec/MTBF_sys grows linearly with node count. | Give the formula and the inputs. | should-fix |
| 15 | 0, 1 | [measured] "2.86 GB at 1B/4 ranks vs 3.00 GB (5%)"; jobs "33037-33039 vs 33042/33043"; t_step "0.699 s" | 2.86 GB and these job IDs are not in `measured_all.md`. The file gives "baseline 0.70 s/step" (`:7`), not 0.699. | Drop the [measured] tag or add these to `measured_all.md`. The [derived] numbers do not change materially at 0.70: k = 8 gives 43.75%, k = 70 is unchanged. | should-fix |
| 16 | 6, 8 #6 | "PG init above 16 nodes (unmeasured)" | PG scaling was measured only at 1-2 nodes. The 4-16 node points are queued (`measured_all.md:11-12`). | "unmeasured above 2 nodes" | should-fix |
| 17 | 2 row 7, 7 | Filesystem outage: "FT is FS-independent", 0 work lost, and restart is unaffected. | A trainer restart spawns processes and imports them from Lustre (`measured_all.md:5`). Data loading and the durable checkpoint also use the shared FS. A crash during an FS outage therefore cannot restart, even though the replica is in tmpfs. | Limit the claim: training continues as long as nothing crashes. | should-fix |
| 18 | 2 row 4/6 | Hang → "timeout + T_rec" | The default timeout is not stated. `nccl_timeout = 3600` s (`config.py:376`), passed at `train.py:108,113`, so detection takes about 1 h by default. | | should-fix |
| 19 | 2 row 9, 4, 9 | A ring distance d ≥ 2 makes row 9 survivable ("row 9 was unwinnable in all prior drafts"). | d ≥ 2 only moves the fatal pair from (i, i+1) to (i, i+d). For two independent failures the number of fatal pairs is still N. It helps only if failures correlate with index adjacency, and the plan gives no evidence that they do. | State the correlation assumption. | should-fix |
| 20 | header, 2 rows 1/10 | "self-contained, needs no prior context" | The torn-safe protocol is cited as "v2 §6" and is not described in v4. | Inline the protocol. | should-fix |
| 21 | 5.3, 5.4 | Up to 375-422 GB of `/dev/shm` is used per node. | Nothing shows that the job's memory cgroup allows this. tmpfs pages are charged to the cgroup that writes them. Also, Slurm epilogs often clean `/dev/shm`, which matters across a requeue. Unverified. | Add this to §8 with a test. | should-fix |
| 22 | 2 row 11 | "No spare injection possible (`@lru_cache`, `slurm.py:87,97`)" | The cache is per-process, and a new process recomputes the address and port. See #4: a spare node inside the allocation has already been used in a measurement. | | should-fix |
| 23 | end | "Citation check: 5 of 49 references don't hold" | The footer is wrong. `xllm/config.py` has 444 lines, and the `config.py` citations are correct. | Remove the footer. | should-fix |
| 24 | 0, 3, 4, 7 | "8 GPUs/node, hard-coded (`slurm.py:149` assert)" | The assert is `0 <= local_rank < 8`, which is an upper bound, not a fixed 8. 8 GPUs/node is the config default `SlurmConf.gpus_per_node = 8` (`config.py:64`) and the hardware fact in `measured_all.md:13`. | | minor |
| 25 | 3 | HSDP: with rs ≤ 8 all replicas sit on one node (`utils.py:133-134`). | That holds for tp = 1 only. For tp > 1 the mesh is `view(fss, rs, tp)` (`distributed/utils.py:141-142`), so a replica group spans rs·tp consecutive ranks. Co-location needs 8 % (rs·tp) == 0. | | minor |
| 26 | 3 | "`fss = 1`, TP-only, is legal" (`build.py:73,75`) | fss = 1 is legal only if cp = 1 (`build.py:128`). It means full replication across dp (DDP-like), not "TP-only". | | minor |
| 27 | 1 | B_own,eff = 4.2 GB/s; B_peer,eff = 6.0 GB/s; C_peer = 0.8–1.2 s | 6.04/(0.8–1.2) = 5.0–7.6 GB/s, so 6.0 is one point in that range. These are training-time costs per copy, not transfer bandwidths. C_peer subtracts "own copy only" (sync or async not stated in `measured_all.md:8`) from the async full copy. From the sync numbers it would be 1.5–1.9 s. | | minor |
| 28 | 3 table | Row 11 is listed as mode-sensitive. | Its own row reads "same" in every mode. | | minor |
| 29 | 7 M1.5 | Grace needs ~15 s at 1T/512. | t_step + C_total = 2.73 + 9.5 = 12.2 s. Own copy only: 2.73 + 1.45×3.9 ≈ 8.4 s. Also, ranks get SIGTERM at different times, and the plan does not say how they agree on which step to stop at. | | minor |
| 30 | 8 #9, #11 | "#9 is not workspace-citable"; "#11 covered by #1 + #10" | #9 can be cited at `torch/optim/adam.py:178-184`. The 0.57 s/GB in #11 comes from the measured restore of 3.42 s / 6.04 GB (`measured_all.md:3`), not from #10. | | minor |
| 31 | 5.4, 5.2, 6 | S_node 187.3 GB; 0.25 GB buffer; C_total 9.5 s | Rounding: 1e12/512×12×8 = 187.5 GB; 256 MiB = 0.268 GB; 2.45×3.906 = 9.57 s. The 504 GB `/dev/shm` figure does not say GB or GiB. | | minor |
| 32 | 0 | `utils.py:141-158` (restart counting) | The range cuts off mid-function. | `get_n_restarts` runs to `xllm/utils.py:167`. | minor |
| 33 | all | torch library lines | They come from torch 2.13.0. The runs used torch 2.12 (`measured_all.md:1`). | | minor |
| 34 | 2 row 12 vs 4 | The data loader is "resumable" from the replica. | This contradicts §4, which leaves the data-loader state out of the replica (same root cause as #2). | | minor |

## Arithmetic that reproduces

- §6: k = 8 overhead is 2.45/(8·0.699) = 43.8% (band 41.1–46.5%).
- §6: k at 5% overhead is 70.1 (band 65.8–74.4).
- §6: projected t_step and C_total ∝ P/W: 2.73/9.57, 1.37/4.79, 0.68/2.39, 2.05/7.18, 1.02/3.59.
- §5.4: params/GPU, S_rank, S_node, host need and GPU state at 16 B/param for all 5 configs.
- §5.3: fit limits of 1.75B and 2.625B.
- §5.1: 31.25, 35.2 and 39.1 GB.
- §3: HSDP rs = 8 verdict (62.5 GB for 1T/2048; the others are over 80 GB).
- §0: 0.7% error.
- §1: async hiding 21–23%.
- §2: restore 0.566 s/GB × 23.4 GB = 13.3 s; disk read 187.5/5 = 37.5 s; 80.8 − 13.3 = 67.5; 80.8 − 13.3 + 37 ≈ 104.
- §5.2: 23.4/(80 − 31.2) = 48%, 23.4/(80 − 35.2) = 52%.

## Citations that checked out

- `slurm.py:20-23`, `:28-30`, `:87,97`, `:88-94`, `:98-105`, `:88-105`, `:116-121`, `:139`
- `initialize.py:60`, `:79,89,103,119`. These pass backend=None (`train.py:113`), so the groups inherit the world's NCCL backend.
- `xllm/utils.py:105-107`, `:176`
- `xllm/distributed/utils.py:115`, `:133-134`, `:138`
- `checkpointing.py:371`
- `config.py:147`, `:352`, `:383-387`, `:418`
- `optim/optimizer.py:5,26-32`
- `train.py:167,191`, `:432-453`, `:436-437,459`, `:438-442`, `:456`, `:481`
- `transformer.py:540,601`; `gekko.py:220,336` (correct but incomplete, see #11)
- `modules/context_parallel/mappings.py:31,46,61,80,117,135`
- `models/fused_blocks/mha.py:118-120`
- `modules/moe/expert.py:72-74` (the expert split itself is at `:70`)
- `measured_all.md:8`
- `refs/nvidia_resiliency_ext/fault_tolerance/launcher.py:1659-1660`
- torch `_flat_param.py:935`, `:1342-1357`, `:1529-1530`; `_runtime_utils.py:757-764`, `:1046-1048`. The text is correct, but these are FSDP1 lines and do not apply to the default path (see #1).
- Zero in-tree matches confirmed for elastic/spare/rendezvous/rejoin and Xid/nvidia-smi/dmesg, and there is no EP group.
