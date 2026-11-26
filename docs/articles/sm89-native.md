# Native SM89 CUDA: routing, calibration and qualification

An architecture-specific image is a starting point, not a measured acceleration.
The Windows wrapper targets Ada Lovelace compute capability 8.9 with CUDA 13.4.
A route must fit tensor layout, shape, registers, memory behavior and numerical
contract. The device calibrator measures candidates and only overrides measured keys.

## Evidence and limits

The 512-token follow-up compared v0.1.3 to candidate defaults on two repetitions:
32K −0.88%, 64K +1.85%, 128K +3.07% decode. The earlier 128-token runs regressed.
The tested bucket/calibration flags did not establish gains over candidate defaults.
Flat-graph work uses bounded resource envelopes and safe fallback; it remains
experimental. Do not equate synthetic graph tests with whole-model performance.

## Qualification contract

Freeze source/binary/model hashes, toolchain, driver, KV format, prompt, context,
sampling and clocks. Alternate A/B runs and retain warmups, raw outputs and
repeated measurements. Check greedy outputs, retrieval, latency, throughput,
VRAM, failures and unsupported routes; isolate memory-transfer and kernel tests.

[Development workflow](../rtx4090.md) · [54-case test plan](../performance/rtx4090-sm89-ab-assignment.md) · [Long-context evidence](../performance/rtx4090-long-context-ab-2026-11-25.md)

Expanded public articles: [English](https://spk-c.github.io/ninfer-all-rtx4090/sm89-native.html) · [Deutsch](https://spk-c.github.io/ninfer-all-rtx4090/sm89-native-de.html).
