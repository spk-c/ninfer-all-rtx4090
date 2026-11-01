# RTX 4090: stability, coding-cache validation and measured profiling

Audit: **2026-10-31**; branch `feature/rtx4090-upstream-stability-cache-profiling`;
[Draft PR #10](https://github.com/spk-c/ninfer-all-rtx4090/pull/10).
Starting revision `8b66e253`; current implementation and fresh measurements:
**`b6c0e4d2416542ac32e2f474077cb9049852bb47`**. Its complete 372-test rerun finished: **340 passed, 32 skipped, 0 failed** in **2087.68 s**.

The [detailed audit](../../profiles/bench/rtx4090-audit/audit.md) retains all source
SHAs, upstream decisions, rejected experiments, failed campaigns, earlier results
and reproduction commands. Current small evidence:
[graph samples](../../profiles/bench/rtx4090-audit/graph-samples-b6c0e4d2.csv),
[cache cases](../../profiles/bench/rtx4090-audit/cache-summary-b6c0e4d2.json),
[paired A/B](../../profiles/bench/rtx4090-audit/graph-ab-b6c0e4d2.json),
[hardware counters](../../profiles/bench/rtx4090-audit/kernel-counters-b6c0e4d2.json),
[phase CUDA APIs](../../profiles/bench/rtx4090-audit/cuda-api-phases-b6c0e4d2.json).
Large raw traces, SQLite, request/reply logs and models stay local.

## Environment and implemented fixes

Windows 10 Pro 19045; RTX 4090 24,564 MiB WDDM, driver 617.42; Ryzen 9 9950X3D,
about 61.6 GiB RAM; CUDA 13.4/nvcc 13.4.92; VS 2022/MSVC 14.44; CMake 3.29.2,
Ninja 1.12.0, explicit Python 3.11.9. Native SM89 Release, freshly rebuilt.
Artifact: `C:\ninfer-all\models\Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer`,
v3, ID `59ac51f9c2774978b3688f36043c56f8`, 15,017,456,896 bytes.

`origin` uniquely names the requested fork. Fetched it and configured local
`checkout.defaultRemote=origin`; the correct branch was already active. The
untracked user script, master and release tag `rtx4090-v0.1.0` are preserved.

Three production corrections:

- Windows hybrid-expert captured H2D payloads blocked the queue needed by the
  supervisor DMA releasing a GPU wait. Exact mapped-host copy kernels now carry
  the three small exchange payloads. Byte/guard/replay and numerical expert tests
  pass. Up to three copy kernels replace H2D nodes; real offloaded-model throughput
  cost is unmeasured. The target dense model does not execute this native expert path.
- IOCP concurrency1 blocked a reader handed to a worker while its former caller
  stayed runnable. Removed that artificial dispatch limit, retaining bounded
  buffers and serialized batch ownership; the old-code regression fails without hanging.
- Default-stream padding clears could run after nonblocking weight uploads and
  erase selected bytes (observed0 instead of 49). Zeroing now precedes H2D on the
  rank's transfer stream, with guarded completion including zero-only ranks.
  An independent CPU panel oracle and 256-round stress reproduce the old bug;
  four affected tests and ten stress runs/2560 rounds pass after correction.

Windows PCI/sysfs and file-truncation fixtures were corrected. The Linear
concurrency harness executes supported A16 on SM89 while retaining A8/A4 TMA on
supporting devices. No test was disabled; this does not qualify Blackwell TMA.
Existing upstream allocation rollback, listener reuse and private Chat anchors
were not duplicated. Divergent startup-budget, pre-SSE admission and legacy-export
changes were not transplanted. The detailed report reviews all seven source commits.

## Actual coding-cache workload

Real local Chat HTTP with coding/tool histories, **not the Qwen Code client
executable**. DFlash2 K5 plus product-default copy width 15, rk4v4, FP16 GDN,
two lanes, context/KV 32768, requested prefill 1024. Separate Hybrid 256-MiB Host
budget and `--no-prefix-reuse` servers; five cold/warm pairs per transport plus
20K-token history, tool turns, concurrent sessions, invalidation and cancellations.
Both 48-scenario arms pass all assertions and stop cleanly.

| Metric, n=5 | Hybrid cold min/median/max | Hybrid warm min/median/max | Disabled repeat min/median/max |
| --- | --- | --- | --- |
| SSE client TTFT ms | 1057.131 / 1061.198 / 1068.116 | 22.477 / 33.207 / 33.677 | 980.321 / 994.597 / 1023.641 |
| Aggregate response ms, not TTFT | 1223.788 / 1242.232 / 1285.582 | 205.397 / 211.536 / 217.087 | 1150.216 / 1179.998 / 1181.958 |

Actual Engine work changes from 3536 computed/0 reused to **7 computed/3529 reused**.
SSE prefill is 1044.659–1050.426 ms cold versus 20.969–21.605 ms warm. All ten
cold/warm generated texts match within each mode.

**Costs and limits:** Hybrid cold median SSE is 63.992 ms (6.42%) slower than the
disabled cold arm; aggregate cold is 63.563 ms (5.39%) slower. The mode groups run
sequentially, so ordering/resource effects are not isolated. Across modes,
**7 of 44 identical payloads still generate different greedy text**; four later
long-session prompts also differ as replies evolve. Universal cross-mode output
identity is not established. Fixing upload ordering does not remove these differences.

Tool extensions reuse 3529/3580 tokens and compute 58/65; repeats compute 7.
Omitting `prompt_cache_key` still hits: Hybrid content addressing shares identical
prefixes across keys; keys are not isolation namespaces. Changing the leading
system prompt gives 0 reused. Concurrent ALPHA/BETA replies match their own fresh
references. Long continuations compute 61/61/45/30 and reuse
20606/20660/20714/20752 tokens at 63–77 ms TTFT.

Client disconnects settle in 76–310 ms in these cases. Full queue returns 429;
queued SSE reports `request_queue_timeout` after 2.016–2.042 s. Subsequent requests
work and final request counts are zero. Native rk4v4/FP16-GDN DFlash2 Host/Device
and persistence tests pass again, including exact token IDs, partial Host-tier
restore, incompatible identity, active stop and abandoned saves.

Board memory spans 17167–17172 MiB Hybrid versus 16817–16826 MiB disabled,
a341–355 MiB difference. This is WDDM device-wide usage, not certified cache
allocation. Host RSS was not measured;256 MiB is the configured budget. Chat has
no `cache_prompt:false` control here; that extension belongs to text completions.

## CUDA Graph capacity and startup

Five successful neural-K5/copy-width 0 starts at each capacity on b6c0e4d2:

| Context/KV | Graph delta MiB, all 5 | Allowance MiB | Board peak MiB min–max | Engine startup s | CUDA free after startup |
| --- | --- | --- | --- | --- | --- |
| 8192 | 34 | 288 | 16002–16006 | 3.9 | 8.46 GiB |
| 32768 | 42 | 384 | 16426–16429 | 3.9 | 8.04 GiB |
| 65536 | 50 | 480 | 16994–16994 | 3.9–4.0 | 7.49 GiB |

Earlier source1922c64a/default-copy-width 15 campaigns observed86/106/126 MiB
and allowances576/768/960 MiB, ten starts per capacity. All60 graph-start records
are retained across sources/configurations; none failed. Removing copy proposals
is a capability tradeoff, not a free optimization. No startup OOM occurs, but
these probes do not fill a64K prompt. Requested100-ms queries can miss peaks.
Graph deltas are device-wide; startup includes loading, not isolated capture time.
Direct-I/O payload loads repeat in fresh processes; disk/driver cache state was
not reset or measured. CUDA free memory and NVML board usage have different accounting.

## Systems observations and real kernel counters

Required **Nsight Systems 2026.1.3.425** was used again on b6c0e4d2. It warns that
CUDA 13.4 is unsupported (13.3 collection libraries). An earlier installed2026.5.1
supplement removes that warning but both versions report incomplete events.
Complete CPU execution time, causal GPU idle intervals and full utilization are
**not measurable from these captures**. Board NVML percentages average earlier
sample windows and do not resolve instantaneous half-second phases.

Current pp2048+tg128, one warmup, one measured repetition, neural K5, copy width 0:

| Captured ms | Prefill | Decode |
| --- | --- | --- |
| NVTX host phase wall | 555.591 | 531.388 |
| Captured kernel elapsed union | 549.964 | 508.917 |
| Outside captured CUDA activity; not proven idle | 5.480 | 22.331 |
| Host submit wall | 222.212 | 23.360 |
| Host wait wall | 333.269 | 507.543 |

Decode records 25 Graph API calls totaling 22.373 ms and 26 stream synchronizations
at 507.320 ms. API elapsed time includes driver waiting, not CPU instruction time.
Captured IQ3_S/IQ4_XS/IQ3_XXS summed-duration shares are28.3967%/18.2503%/16.3734%,
not whole elapsed GPU percentages. New versus earlier separate captures show
prefill+4.850 ms and decode-5.156 ms; neither is isolated version A/B evidence.

Six fresh Nsight Compute 2026.3.1 launches, nine-pass kernel replay, default cache
flushing, unmodified clocks; different real shapes, **not format-versus-format A/B**:

| Format | Prefill us / registers / occupancy% / GB/s | Verification T6 us / registers / occupancy% / GB/s |
| --- | --- | --- |
| IQ4_XS | 501.600 / 254 / 16.665 / 83.745 | 32.768 / 134 / 23.821 / 853.535 |
| IQ3_S | 577.472 / 254 / 16.665 / 66.775 | 30.208 / 128 / 30.961 / 752.479 |
| IQ3_XXS | 361.120 / 238 / 16.660 / 70.240 | 19.680 / 101 / 29.486 / 633.678 |

Prefill is limited to one CTA by **both registers and shared memory**. Verification
has much higher DRAM throughput. These are individual-launch counter results,
not whole-model utilization. Earlier failed NCU filters and compressed-report
extension assumptions were corrected from actual kernel names and reports.

The repetitive corpus earlier accepted 120/120 context-copy drafts in eight
16-token rounds. Current neural K5 accepts 103/120 in 24 rounds plus one fallback.
Earlier profiled562.70 versus current 241.115 tok/s describes different paths and
source captures, not a code-version speed comparison.

## Unprofiled alternating Graph/Eager A/B

Identical80-function coding prompt,128 requested output tokens, K5 plus default
copy width 15, rk4v4, FP16 GDN, 8192 context. Five fresh-process pairs; leading arm
alternates. All ten processes exit0; all five generated texts match exactly.
Desktop applications remain open; no inference/profile/test competes with the trials.

| Pair | Eager tok/s | Graph tok/s | Graph/Eager |
| --- | --- | --- | --- |
| 1 | 163.0 | 190.6 | 1.16933 |
| 2 | 163.2 | 190.8 | 1.16912 |
| 3 | 163.8 | 190.8 | 1.16484 |
| 4 | 163.5 | 190.5 | 1.16514 |
| 5 | 163.5 | 190.0 | 1.16208 |

Paired median advantage **16.514%**, range 16.208–16.933%,5/5 positive. Graph
Engine startup costs an extra **0.7–0.8 s** in every pair. Earlier1922c64a results
included184.7 tok/s and a13.382% worst pair; this unexplained adverse result is
retained. This compares **existing optimized Graph against Eager fallback**,
not a newly implemented fork speedup. No experimental kernel default is enabled.

Whole-process latency goes the other way for this short fresh-process workload:
Graph is 0.239–0.322 s slower in every current pair (median 0.278 s, paired
median+5.10% wall time). Faster decode therefore does not mean faster one-shot
CLI execution. The older campaign also had0.203–0.312 s higher Graph wall time.
Long-lived serving amortizes startup; no global default is changed.

## Verification status and remaining scope

Full native rebuild succeeds. Historical complete campaigns:
337 pass/32 skip/2 fail, then338 pass/32 skip/1 fail. The latter exposed actual
weight corruption and was not called a pass. Earlier incomplete campaigns found
the Windows deadlocks. The post-fix **372-test campaign completed: 340 pass / 32 skip / 0 fail**, 2087.68 s; finished JUnit and the full log agree.
All25 affected local Python tests pass. The separate
[staged-backport CI](https://github.com/spk-c/ninfer-all-rtx4090/actions/runs/41231059638)
passes20 Python tests and Windows PowerShell parsing for unchanged tooling;
main CI intentionally skips Draft PRs and is not counted as passed.

Corrected fixture/tool issues include Windows paths/truncation, unsupported TMA
selection, an assumed fixed cache frontier, a persistence-backend mistake, mutable
recorded request histories, numeric Windows kernel names and PowerShell exit-code
shadowing. Failed eager-module/alternative-DMA experiments were reverted. The
report retains every failure and changed measurement configuration.

The 32 optional artifact/architecture skips are not passes. MSVC GNU-linker `--wrap` fault-injection subcases are compiled out and additionally unverified; supported VMM paths did execute. Windows single-card evidence
does not certify Linux, other artifacts, multiple physical GPUs, vision or
Blackwell TMA. Real offloaded-model throughput and Host RSS remain unmeasured.
Useful next work: jointly lower MMQ register/shared-memory pressure; examine Vec
traffic separately; isolate cache-mode numerical variations; qualify graph budgets
across more models; resolve Windows trace coverage. Every candidate still needs
independent numerical oracles and matched alternating whole-inference A/B before
any default change. Master and release remain untouched.
