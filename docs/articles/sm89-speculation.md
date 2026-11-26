# DFlash2 and MTP on SM89: draft width and memory

Speculation trades proposal and verification work against committed tokens.
DFlash2 and MTP are alternative, model-dependent backends. A larger draft
window may cost extra time and memory; acceptance is not throughput.

Exact published v0.1.4 Engine benchmark: Huihui IQ3_S, RTX 4090, 2048 prompt
tokens, 128 timed decode tokens, 32K capacity, rk4v4, GDN FP16, 2 repeats,
1 warmup, n-gram width 0.

| Backend | Decode tok/s | Acceptance |
|---|---:|---:|
| none | 70.38 | — |
| DFlash2 K3 | 201.93 | 93.07% |
| DFlash2 K5 | 257.13 | 85.83% |
| DFlash2 K7 | 326.41 | 88.00% |
| DFlash2 K9 | 269.33 | 74.50% |
| MTP K3 | 200.31 | 89.42% |
| MTP K5 | 242.59 | 85.83% |

K7 wins this fixed short corpus, not necessarily coding or long context.
Bonsai 2.5 tests use different artifacts and n-gram width 15; no direct
quality or acceptance ranking. Measure same workload and cache state, verify
output quality, reserve VRAM for drafter, workspaces, state and desktop.

[Exact benchmark report](../performance/rtx4090-v014-bonsai-and-draft-2026-11-26.md) · [Launch planner](https://spk-c.github.io/ninfer-all-rtx4090/configurator.html) · [English article](https://spk-c.github.io/ninfer-all-rtx4090/sm89-speculation.html) · [Deutsch](https://spk-c.github.io/ninfer-all-rtx4090/sm89-speculation-de.html).
