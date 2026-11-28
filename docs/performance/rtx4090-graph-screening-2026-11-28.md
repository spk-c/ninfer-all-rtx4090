# RTX 4090 CUDA Graph allowance: preliminary screening (2026-11-28)

Related: [#56](https://github.com/spk-c/ninfer-all-rtx4090/issues/56), merged telemetry correction [#64](https://github.com/spk-c/ninfer-all-rtx4090/pull/64).

**Decision: keep current conservative SM89 defaults.** This is a retrospective audit of the actual phase-7 Windows RTX 4090 logs, not a new experiment or a qualification of the v0.1.8 archive.

Candidate source `1b5313ad`, binary/model hashes and NVIDIA driver 621.40 are recorded in [the aggregate JSON evidence](rtx4090-graph-screening-2026-11-28.json). Each cell used one independent server start, DFlash2, RK4V4, prefill 1024, n-gram disabled, configured 32,768 context and concurrent synthetic 5,303-token prompts. All replies were exactly OK.

| Concurrency | Draft K | Allowance MiB | Net observed MiB | Physical KV tokens | Ready free MiB | Exact replies |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 256 | unmeasured | 405120 | 881 | 1/1 |
| 1 | 5 | 384 | 49.62 | 397632 | 815 | 1/1 |
| 1 | 7 | 256 | 44.00 | 404800 | 847 | 1/1 |
| 1 | 7 | 384 | unmeasured | 397312 | 860 | 1/1 |
| 2 | 5 | 256 | 106.00 | 382272 | 818 | 2/2 |
| 2 | 5 | 384 | 106.00 | 374784 | 767 | 2/2 |
| 2 | 7 | 256 | 114.38 | 381632 | 775 | 2/2 |
| 2 | 7 | 384 | 250.01 | 374144 | 792 | 2/2 |
| 4 | 5 | 256 | 231.94 | 336512 | 598 | 4/4 |
| 4 | 5 | 384 | 398.04 | 329024 | 602 | 4/4 |
| 4 | 7 | 256 | 232.00 | 335232 | 672 | 4/4 |
| 4 | 7 | 384 | 311.82 | 327744 | 641 | 4/4 |

All 12 groups completed, but two legacy zero net deltas are ambiguous and one positive delta (C4/K5, 398.04 MiB) exceeds the 384 MiB allowance. Physical KV capacity is not usable logical context. The net observation is device-wide and cannot isolate CUDA Graph allocations. No default reduction follows from these results.

## Remaining qualification

- Use the exact published v0.1.8 binary and immutable model hashes with idle GPU, record driver, graph coverage and actual request/peak memory evidence.
- Repeat default and explicit candidate allowance starts; cover DFlash2 K5/7/9/11/15, fixed/adaptive MTP, C1/2/4, 64K/128K/176128 contexts, n-gram, vision, flat graphs and graph buckets where supported.
- Require matched prompt/output correctness, concurrent prefill and overlap, no allocation failures, repeated starts and a conservative margin. Do not equate startup screening with safe default qualification.
- Keep the Windows host-KV rejection and fallback unchanged. Compare Windows and Linux separately.

The GPU is in active production use at the time of this audit. No competing GPU benchmark was launched. The CPU analysis now records artifact identity and rejects incomplete or over-budget data as complete startup screening. The full qualification remains open.
