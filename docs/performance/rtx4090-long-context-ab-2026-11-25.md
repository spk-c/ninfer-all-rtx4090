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
