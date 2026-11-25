# SM89 long-context A/B and architecture migration evidence (2026-11-25)

## Reference: verified v0.1.3

Source 8de6beb22c545003e90c404226f066ddc4f56271, RTX 4090,
Huihui Qwen3.8-27B GSQ-RCO IQ3_S Vision DFlash2, RK4V4, FP16 GDN,
prefill chunk 1024, adaptive K5, N-gram disabled. One measured repetition per context
(no discarded warmup). The 65,536-token source fixture was repeated for 128K.
The corpus SHA256 is 783f564e2bc1dc4d4cf3b4d28c64977d87b4910ed9b1db4ad96f1aa1ab0231b2.
The benchmark consumes one seed output plus 128 measured decode tokens.

| Max context | Prompt tokens | Prefill tok/s | Decode tok/s |
|---:|---:|---:|---:|
| 32768 | 32512 | 3538.789033 | 260.0527663 |
| 65536 | 65280 | 3227.832747 | 257.509728 |
| 131072 | 130816 | 2722.460259 | 233.2522174 |

These are baseline-only samples, not speedups. Raw JSON and logs remain on the
SentinelX test machine. The packaged executable's runtime report identifies
CUDA 13.5; the archive build metadata labels the package CUDA 13.4.

## Candidate

Use scripts/ab-sm89-long-context.ps1 with separate baseline/candidate executable
directories, the exact clean candidate source SHA and the same model.
The script records binary/model/corpus SHA256, alternates AB/BA, rejects
incomplete or mismatched measurements, and keeps every raw report.

Experimental flags are process-local and disabled by default:
- NINFER_SM89_STARTUP_CALIBRATION=1: bounded graph warmup latency priors
  at short representative frontiers; no synthetic acceptance data or long-context claim.
- NINFER_SM89_ATTENTION_BUCKETS=1: fixed split topology for the existing cyclic
  draft attention, retaining BF16 cyclic cache and both masks; not the complete
  donor attention migration.

Local tests: startup calibration schedule/prior passed; full prefix index tests
passed; Qwen3.5 graph, context planning and hybrid orchestration units compiled;
real SM89 sliding-window attention graph test passed 16 replays through 128K
with exact constant-value and inactive-tail oracle. The complete donor graph
suite, causal/sparse/dense attention migration, physical startup fit and live
request preemption/replay are not yet qualified. Donor mainline only supports
Linux RTX5090 (sm120a) and DGX Spark (sm121a), and changes KV/layout interfaces;
it must not replace the RTX4090 Windows fork by a blind merge.

No stable tag or master changes. No candidate speedup claimed before paired measurements.

## Completed paired A/B: 859d76b7

All 12 runs finished successfully (two alternating AB/BA pairs at each context).
Both experimental flags were enabled on the candidate. The exact benchmark
binaries and model/corpus hashes are recorded under
profiles/bench/rtx4090-neroued-20261125/manifest.json, alongside all 12 raw
reports and logs. The repeated corpus in this paired experiment has the same
token count as the reference fixture but uses different trailing newline
serialization; use its manifest hash, not the earlier single-run file hash.

| Context | Baseline decode tok/s | Candidate decode tok/s | Decode delta | Prefill delta |
|---:|---:|---:|---:|---:|
| 32768 | 241.278896 | 233.079614 | -3.40% | +6.11% |
| 65536 | 253.614670 | 245.273506 | -3.29% | -1.31% |
| 131072 | 227.385206 | 221.188425 | -2.73% | -1.83% |

The regression is observed for this workload; two pairs and a 128-token
decode are not enough to isolate causality. Upstream changes to route profiles
(68 vs 130 routed keys), build/runtime differences and the experimental flags
are confounded. Both flags remain opt-in. No release promotion.

A five-arm follow-up (baseline, candidate defaults, buckets only, startup
calibration only, both) uses scripts/ab-sm89-factorial.ps1 with the exact
same original candidate binary, same corpus/model, 512 decode tokens and
reversed arm order on alternate repetitions. The results must be assessed
before choosing a performance optimization or promoting a release.

## Five-arm follow-up: 859d76b7, 512 decode tokens

All 30 runs completed (two repetitions, reversed arm order).
Raw reports: profiles/bench/rtx4090-neroued-factorial-20261125.

| Context | v0.1.3 | Candidate default | Buckets | Calibration | Both |
|---:|---:|---:|---:|---:|---:|
| 32768 | 262.78 | 260.46 | 260.48 | 260.42 | 259.88 |
| 65536 | 245.67 | 250.22 | 249.90 | 250.25 | 250.36 |
| 131072 | 214.29 | 220.86 | 220.98 | 220.42 | 220.26 |

Candidate default: -0.88%, +1.85%, +3.07% decode; prefill almost unchanged.
Neither experimental flag demonstrates a reproducible improvement over
candidate defaults; both stay off. The first short-decode regression does
not generalize to the longer decode workload. These are two repetitions,
not a statistical guarantee. The corpus repeats a fixture and is not a
representative production workload.

## Subsequent experimental migration

Dense BF16 paged draft attention now uses canonical bucket resource promises
when NINFER_SM89_ATTENTION_BUCKETS=1, clamped to the actual logical context
capacity. Planning and execution use the same envelope. The standalone
SM89 graph-slice test passed 28 replays with two alternating physical page
values, zero-length through 196608-token frontiers, T=2 and T=12, and an
analytic nonconstant oracle with exact inactive tails. Qwen3.5 context,
decode, graph and startup planning translation units compiled.

NINFER_SM89_PHYSICAL_STARTUP_CALIBRATION=1 (in addition to the startup
calibration flag) optionally uses distinct preallocated KV pages, including
the target and any paged draft backend, at 64/2112/8192/32768 tokens when
capacity permits. Admission checks account for earlier probes, available
pages and per-address limits; the cap is 1024 pages per pool and the soft
probe time budget is five seconds. Pages are released after startup. This
provides physically backed synthetic latency priors, not an acceptance
estimate or a full physical cost-model fit. Host budget tests passed;
real-model qualification is still required.

The full donor flat graph suite, remaining causal/sparse attention migration
and live request preemption/replay remain unimplemented.

## Model-integrated flat draft readouts (experimental)

The single-rank neural Qwen3.5 DFlash and DFlash2 graph capture can now
collect value-owned attention operands in a first pass, prepare independent
kernel-slice variants against a borrowed arena with the original base and
offset, and record the readouts inline in a second parent capture. Runtime
selection patches the resident parent when crossing canonical buckets within
a target graph profile. The second capture verifies operand pointer keys and
workspace identity/offset. The parent always contains the largest safe
envelope; a patch failure reinstantiates it and disables that profile's
patch path. Profile switches requiring slices instantiate the exact parent.
Startup optionally exercises each prepared bucket and reports profile/site/
replay/fallback counts.

Enable NINFER_SM89_FLAT_DRAFT_GRAPHS=1 together with
NINFER_SM89_ATTENTION_BUCKETS=1. Both remain disabled by default.
This path deliberately excludes copy rounds, DSpine and multi-rank execution.
Target verification still uses its existing measured graph profiles, not the
donor target flat suite. The dense and sliding draft readouts share this
recorder; proposal/verify/decision/commit graph separation is still open.

The draft, decode and graph preparation translation units compiled with
MSVC/CUDA 13.4 SM89. The standalone two-site CUDA test was compiled and linked;
GPU execution and whole-model qualification have not yet occurred because
the earlier long-context benchmark campaign is using the RTX4090. No
performance claim for the flat path until measured.

## Completed 2d37ef0a campaign

All 54 runs across ten configurations succeeded. Full evidence:
profiles/bench/rtx4090-neroued-tuning-20261125. At 176128 and 196608
tokens the two-pair default candidate decode gains were +1.89% and +3.07%.
Physical startup completed 8/8 probes at 32K and 6/6 at 128K, no resource
skips; these are synthetic-token latency probes on distinct physical pages,
not a fitted donor physical model. Single-pair draft tuning favors K7 at
32/64/128K (294.10, 284.61, 241.52 tok/s), while K9 gains most relative to
its own old binary. The 2048-token prefill setting improves prefill over
1024, with nearly the same prefill rate as 4096. One-factor single-pair
results and repeated corpus are not a universal preset; no default change.

The 1de2e5c6 flat draft two-site GPU test subsequently passed six actual
replays, borrowed-scratch identity, changed operand rejection and fallback
reinstantiation. c01528e3 adds optional causal target verification slices
for resident single-rank Qwen3.5 masked neural rounds; the smaller resource
promise includes the physical verification width and falls back to the
original envelope if the route family changes. NINFER_SM89_FLAT_TARGET_GRAPHS=1
also needs the two flat draft flags. Target/execution/decode/graphs compiled.
Whole-model exact generated-token comparison and benchmark remain required.

0e52018e adds one-shot, opt-in hybrid context pressure replay at a safe
decode boundary after retained-cache reclaim failed. The prompt is rebuilt
from the original prompt plus all committed tokens before releasing the
active owner; the request retains its existing output session and original
budget, and is queued behind other pending work. This is limited to
ordinary greedy text, no media, grammar, reasoning/control, penalties,
logprobs, ngram archive or previous replay, and requires growing KV leases
and another live lane. NINFER_SM89_LIVE_CONTEXT_REPLAY=1 enables it. The
host eligibility test and engine translation unit compiled/passed. A
two-live-request real-model pressure fixture was added; actual pressure
replay qualification is pending.

The donor's full split proposal/target/decision/commit suite, all other
model families' sparse/MLA/target kernel migration, full physical fit and
general recovery for multimodal/sampled/structured requests are not
completed. The donor targets different architectures and storage layouts.
All experimental flags remain off by default; master and stable tag unchanged.
