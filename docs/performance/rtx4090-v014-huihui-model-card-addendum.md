# Huihui Qwen3.8-27B IQ3_S — v0.1.4 benchmark addendum

For model SHA-256 a410396311ccb27048008a7ccb8a38f91099f6f6a8e10d10bd8baed7248c046a
on NVIDIA RTX 4090 (24 GB), published NInfer RTX 4090 v0.1.4, fixed DFlash2 K7,
native prefill 4096, RK4V4 and N-gram disabled during the Engine sweep:

| Logical context | K5 decode | K7 decode |
|---:|---:|---:|
| 64K | 242.75 tok/s | 272.74 tok/s |
| 128K | 206.39 tok/s | 240.60 tok/s |
| 176128 | 214.87 tok/s | 263.93 tok/s |

Separate exact identifier rename workloads with ~21.4K prompt tokens:
N-gram 0/15 mean decode 256.7/568.1 tok/s; 4/6 exact outputs in both arms.
The short-request concurrent-prefill TTFT was 21.28/1.16 s off/on.
These are workload-specific measurements, not guaranteed inference throughput,
not general coding accuracy or an upstream/other-engine comparison.
The model files and release binaries are unchanged.

[Full methodology and numerical evidence](https://github.com/spk-c/ninfer-all-rtx4090/blob/master/docs/performance/rtx4090-v014-long-context-2026-11-26.md) ·
[Coding report](https://github.com/spk-c/ninfer-all-rtx4090/blob/master/docs/performance/rtx4090-v014-coding-2026-11-26.md) ·
[Opt-in K7 config](https://github.com/spk-c/ninfer-all-rtx4090/blob/master/configs/rtx4090-huihui-dflash2-k7.ini)
