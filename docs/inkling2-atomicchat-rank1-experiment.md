# Experimental Inkling 2 rank-one adapter reference

Scope: CPU numerical oracle only, not a released model or native serving support.
The ISTA-DASLab GSQ-RCO IQ3_S matrix encodings must remain byte-identical.
The AtomicChat GGUF adapter is separate from the base. Source licenses and
redistribution conditions must be reviewed before publishing anything.

## Sources
- Base: ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF, revision f0da1097633dd7c2f5733e4077cff861be122245.
- Adapter: AtomicChat/Inkling-2-Mini-Uncensored-GGUF, revision c5667da36fe35da026fbee394d00547a921e98ab.
- Adapter SHA256: 42914b85c874e7fe249e93f0ac6e8abf08dc0bdf32aa1ec46c72714e64808f12.

The adapter contains 49 pairs of rank-one F32 factors (alpha 1):
24 attention output matrices, 24 expert down-projection banks and one token embedding.
Expert factors are per expert, not one shared residual vector. Core layers use the
same factors on every physical-layer revisit. Embedding factors use the transposed
orientation relative to ordinary linear weights.

## Local reference tests
From the repository root, with the conversion Python environment and gguf package:
```
python test_atomic_rank1_reference.py --adapter PATH_TO_ADAPTER.gguf
python test_atomic_rank1_reference.py --adapter PATH_TO_ADAPTER.gguf --base PATH_TO_GSQ.gguf
python -m tools.reference.inkling2 --gguf PATH_TO_GSQ.gguf --tokens ids.json --rank1-adapter PATH_TO_ADAPTER.gguf --adapter-scale 1 --out adapted.npz
```
Run the same reference command with adapter scale zero and without an adapter.
The 147 synthetic row cases use the actual adapter, not a complete inference.
The full base matrix compatibility checks and end-to-end inference are separate gates.

## Native implementation gates
1. Add strict adapter loading and shape/identity validation; keep the base artifact immutable.
2. Apply embedding row deltas, attention output deltas before post-convolution,
   and routed expert down-projection deltas before the existing expert scales and
   weighted reduction. Apply to all loop visits, and either support or explicitly
   reject unsupported residency routes.
3. Preserve unmodified execution at scale zero; isolate cache identity across
   adapter identity/scale; validate batched prefill and decode, graph replay,
   rollback and speculative verification. Do not silently drop any adapter pair.
4. Compare native outputs with the FP64 oracle and with an independent GGUF
   runtime using the same base/adapter; test DSpark, DFlash2, vision and tool calls.
5. On RTX 4090, measure RK8V4 context capacity, throughput, VRAM and coding
   pass rates (with an unmodified base control). Do not advertise quality gains
   until measured. Prefer an opt-in adapter or adapter-aware artifact over
   requantizing the full expert matrices.

A model package with both drafters and this adapter is not yet a working NInfer
artifact. A runtime change and GPU qualification are required before promotion.

## First integrated candidate (2026-11-28)

- CPU forward: all 48 visits on 4 fixed valid token IDs. Original and scale zero bit-identical; scale one changes logits (max absolute delta 1.828678983). Not a coding benchmark.
- Artifact: 12,966,136,064 bytes, SHA256 18bfef38c79b7b36090d6e3d6a24b859fb605c0796f0d35d15e201ea15863e11. All 169 imported quantized text objects (10,293,138,432 bytes), BF16 token embedding (926,318,592 bytes), and all 98 F32 adapter objects (43,796,736 bytes) verified byte-for-byte against GGUF source. Vision and both drafters included, but not qualified with the adapter.
- Converter: `--source rank1=PATH` embeds factors at fixed scale 1. Loader checks all factor pairs. Initial executor uses the BF16 matrix activation path with original GGUF blocks, one GPU, device experts, no draft and no CUDA graphs. Old runtimes may ignore extra tensors; do not use them with the experimental artifact.
- SM89 standalone CUDA tests: FP32 and BF16 paths, 350,208 values total, maximum absolute error 5.96e-8. MSVC syntax-only checks of executor.cpp and model.cpp passed. Full native compile and end-to-end CUDA inference not yet qualified.
- Scripts: build_inkling2_rank1_experiment.py, verify_rank1_artifact.py, build_rank1_native.cmd, Run-Inkling2-Rank1-Smoke.ps1. The smoke script is prepared, not executed. Build logs under atomicchat-adapter.

Do not promote this artifact until native output, CUDA graph/cache correctness, q8_1/fixed-point path, both speculative decoders, tool calling, vision and real coding benchmarks have passed. The fixed adapter is part of the artifact identity, not a per-request toggle.

## Q8 / CUDA Graphs / speculative decoding follow-up

- Implemented planar Q8_1 rank-one dots and routed per-pair FP32 corrections before existing fixed-point weighted reduction. Base encoded matrices remain unchanged. BF16 activation path remains for wide calls. Graph capture is enabled on the experimental SM89 build. External DSpine and expert offload/parallel remain rejected.
- RTX 4090, same 30-token coding prompt, 512 committed tokens, greedy, RK8V4, 8K: BF16 reference 75.4 t/s, Q8 no graph 86.3, Q8 graph 116.6, original graph 144.1. Q8 graph on/off stdout SHA256 identical. Separate no-drafter 108.3, DSpark K3 240.7 (69.5% acceptance), DFlash2 K7 302.2 (67.1% acceptance); both drafters stdout identical to no-drafter on this prompt. Individual runs, not generalized quality or speed claims.
- 128K, 192K, 256K allocations pass with short prompt and DFlash2; planned totals 14.4, 15.2, 16.0 GiB (no vision). Actual long prefill/retrieval is a separate test. Vision example produces mailbox 24 and sun on right in both models. OpenAI API smoke at 128K with vision, DFlash2, fixed rank1: normal completion 42; forced lookup_weather function call with city Berlin succeeded. Test server stopped.
- Three simple Python tasks, 1024-token output budget, DFlash2 K7: original 18/18 tests, rank1 12/18; merge_intervals truncated at output limit. Small smoke sample only. No full LiveCodeBench or broad coding/behavior evaluation. No public release yet.
- Reproduction scripts: benchmark_rank1_q8.py, benchmark_rank1_drafters.py, benchmark_rank1_capacity.py, benchmark_rank1_coding.py, benchmark_rank1_vision.py, smoke_rank1_server.py, benchmark_rank1_long_context.py. Dedicated experimental PowerShell entry point C:/ninfer-all/Start-Inkling2-GSQ-RCO-Rank1-RTX4090-Experimental.ps1 (128K, port 8082, K7, RK8V4, vision, cap 16384, output default 32768). No modifications to stable Qwen setup.

## Real long-context qualification
- 124,562 actual prompt tokens, 219 completion tokens, all three planted keys recovered, prefill 4871.9 t/s, decode 167.1 t/s.
- Initial 262144 server launch failed due to the previous PowerShell child not yet being gone, not an OOM. The direct-owned-child v2 script fixed the race. At 250,006 actual prompt tokens, 167 completion tokens, all three keys recovered, prefill 3925.6 t/s (63.686 s), decode 142.9 t/s. DFlash2 K7, RK8V4, no vision. Two synthetic retrieval prompts, not general long-context coding qualification.
- License metadata audited at pinned source revisions. Five sources Apache-2.0; ggml vision model card identifies the Apache-2.0 base but lacks separate license metadata. Draft model card at docs/inkling2-rank1-model-card-draft.md. No public release.

## Expanded coding / benign instruction comparison

- Ten matched coding prompts per model, 4096 generated-token limit, RK8V4, DFlash2 K7, greedy. All 20 responses ended normally. Outputs were reviewed and evaluated in a separate isolated container; no generated Python was executed on the Windows host. Both variants passed 10/10 tasks and 62/62 functional checks. Mean per-task decode rate 349.0 tok/s original vs 298.84 tok/s rank1; generated tokens 4147 vs 4296. This is a small local suite, not a standardized benchmark.
- Adjacent integer intervals probe: original returns [[1,2],[3,4]], rank1 returns [[1,4]]. The prompt's wording about touching intervals is ambiguous for discrete integer ranges; the probe is not part of the 62 checks.
- Six matched benign prompts (translation with profanity, non-graphic fictional detective scene, general medical explanation, balanced debate, fictional satire, defensive SQL explanation): both models responded substantively, no refusals observed. This does not establish uncensored effectiveness or general safety behavior.
- HumanEval: official 164-task dataset fetched at pinned commit b9453e8f1865df0a0057a1383767caa400acfbcf; gzip SHA256 b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef. Generation script benchmark_rank1_humaneval.py started for both variants (328 candidates), chat-adapted protocol, temperature 0, RK8V4, DFlash2 K7, 8192 context, 4096 output cap. Candidates only; execution-based evaluation not yet complete. Do not execute generated code outside a robust sandbox. No official pass@1 score claimed.

## Guarded v3 compatibility candidate

The converter now sets text model_type to inkling2_rank1_v1 when embedding fixed rank-one factors. The config parser recognizes exactly the base or v1 model types; loader requires marker and embedding factors to agree and checks every factor pair. Legacy config parser compiled from base commit 466f027 rejected the new marker, new parser accepted base/v1 and rejected unknown v2. CPU-only regression passed; full native rebuild deferred until existing HumanEval process finishes. Guarded artifact created separately; all 1352 objects and 12,965,816,576 payload bytes byte-identical to earlier experimental artifact (payload SHA256 9ec0185096d1899f130582754b7df2797fab674a2388f5584c31913ff63af88c). New file SHA256 bd9fe27df1e29107b5700b33ebc0db2b3bab9a1da6c25a48a85cca76a40e42b6. Old unguarded model remains in place for active benchmark. Do not distribute it.
Vision audit: ggml-org Q8_0 projector SHA256 233287272ec5ac39b4168677aba54a4c024f202410f5629fac1830e1059f8abb; source model Apache-2.0, no separately declared license in pinned ggml card. Unsloth at its pinned Apache-2.0 revision has BF16/F16/F32 projectors only, not the byte-identical Q8_0 file. No public redistribution until provenance and applicable notices are resolved. No weights or user data staged.
