---
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Mini
pipeline_tag: image-text-to-text
tags:
- ninfer
- gsq-rco
- iq3_s
- lora
- experimental
- dflash2
- dspark
---

# Inkling 2 Mini GSQ-RCO IQ3_S + AtomicChat Rank-1 (experimental draft)

**Not a public release. License/provenance and full runtime compatibility remain release gates.**

This candidate combines the original ISTA-DASLab mixed-precision GSQ-RCO IQ3_S text weights with separate fixed-scale AtomicChat rank-one FP32 factors. No text-weight requantization or training was performed. It is not a merged GGUF. The patched NInfer runtime applies the factors to the embedding, attention output, and expert down projections on each physical layer visit.

## Required runtime

The experimental branch `experimental/inkling2-atomicchat-rank1-reference` at base commit `466f027ea5c8397da33ccdf2a9826c82dbb85fcc` plus local modifications, SM89 CUDA 13.4. The guarded candidate uses a distinct model_type. The original source-compiled config parser rejected it, and the modified parser accepted it. All five executable-level smoke cases passed: previous runtime rejects guarded artifact; new runtime rejects unguarded adapter; new runtime accepts normal and guarded models; guarded output matches the previous adapter output on the fixed coding prompt. Do not distribute or serve the earlier unguarded artifact. Only device-resident experts on one GPU are supported by this adapter implementation. DFlash2 and DSpark are alternative drafters, not simultaneous.

Guarded release candidate: `Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-v1-EXPERIMENTAL.ninfer`, 12,966,136,064 bytes, SHA256 `bd9fe27df1e29107b5700b33ebc0db2b3bab9a1da6c25a48a85cca76a40e42b6`. Required text model_type: `inkling2_rank1_v1`. The previously benchmarked experimental file has SHA256 `18bfef38c79b7b36090d6e3d6a24b859fb605c0796f0d35d15e201ea15863e11`; all 1,352 objects and the full binary payload (12,965,816,576 bytes) are identical between the files. The guarded file passed the runtime and API smoke tests; the HumanEval run used the earlier artifact with identical binary payload.

## Provenance and notices

- Thinking Machines Lab: https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/c9d028302cd0f035fdf8e7e91f089a4ce94d2195 (Apache-2.0).
- ISTA-DASLab: https://huggingface.co/ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF/tree/f0da1097633dd7c2f5733e4077cff861be122245 (Apache-2.0).
- AtomicChat: https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF/tree/c5667da36fe35da026fbee394d00547a921e98ab (Apache-2.0).
- DSpark: https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1 (Apache-2.0).
- Vision: https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/tree/0bdc580634ceac4c4d921f452777bf44228de9af (no separate license metadata at pinned revision; source model is Thinking Machines' Apache-2.0 checkpoint; confirm rights/notice inheritance before redistribution).
- DFlash2: https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/tree/4f9088ea66c21f2b99ed657d1941c306b03e84d2 (Apache-2.0).

Include applicable Apache-2.0 license text, copyright and NOTICE material with any distribution. The local audit downloaded the source model cards and available license files. Changes are conversion, embedding of fixed adapter factors, and native runtime integration. The term Uncensored describes the source adapter, not an assurance of any particular refusal or safety behavior.

## Verified results (one RTX 4090, RK8V4, CUDA 13.4)

- 169 encoded text objects, BF16 embedding, and 98 adapter objects verified byte-identical to their source.
- 48-visit CPU reference: scale zero bit-identical to original; scale one changes logits. GPU smoke and isolated CUDA tests passed; not full logit equivalence qualification.
- Same 30-token coding prompt, 512 committed tokens, greedy: BF16 adapter 75.4 t/s; Q8 adapter 86.3 t/s without graphs and 116.6 with graphs; original with graphs 144.1. Separate run: no drafter 108.3, DSpark K3 240.7, DFlash2 K7 302.2 t/s. Outputs of the latter three runs matched byte-for-byte on that prompt.
- Synthetic three-key retrieval: 124,562 prompt tokens, 3/3 keys, prefill 4,871.9 t/s, decode 167.1 t/s; 250,006 prompt tokens, 3/3 keys, prefill 3,925.6 t/s, decode 142.9 t/s. These are one prompt per length, not general long-context or coding scores.
- Vision example and forced structured function call passed. No broad vision, agentic, or multimodal benchmark.
- Earlier three-task coding smoke at 1024 generated-token limit: original 18/18 checks, adapter 12/18 checks; merge function truncated. The subsequent matched coding suites are reported below.
- No broad safety, refusal, or policy evaluation. Benign instruction-following comparison is a limited functional sanity check.

## Release gates

Obtain reproducible committed runtime, fail-closed handling of adapter artifacts for unsupported runtimes or clear compatibility guard, broader coding and inference parity testing, published model license/provenance review, and explicit release approval. No public artifact or runtime release has been created.

## Expanded coding / benign instruction comparison

- Ten matched coding prompts per model, 4096 generated-token limit, RK8V4, DFlash2 K7, greedy. All 20 responses ended normally. Outputs were reviewed and evaluated in a separate isolated container; no generated Python was executed on the Windows host. Both variants passed 10/10 tasks and 62/62 functional checks. Mean per-task decode rate 349.0 tok/s original vs 298.84 tok/s rank1; generated tokens 4147 vs 4296. This is a small local suite, not a standardized benchmark.
- Adjacent integer intervals probe: original returns [[1,2],[3,4]], rank1 returns [[1,4]]. The prompt's wording about touching intervals is ambiguous for discrete integer ranges; the probe is not part of the 62 checks.
- Six matched benign prompts (translation with profanity, non-graphic fictional detective scene, general medical explanation, balanced debate, fictional satire, defensive SQL explanation): both models responded substantively, no refusals observed. This does not establish uncensored effectiveness or general safety behavior.
- HumanEval: official 164-task dataset fetched at pinned commit b9453e8f1865df0a0057a1383767caa400acfbcf; gzip SHA256 b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef. Generation script benchmark_rank1_humaneval.py started for both variants (328 candidates), chat-adapted protocol, temperature 0, RK8V4, DFlash2 K7, 8192 context, 4096 output cap. Execution-based evaluation completed in a separate environment using Python 3.13.5; the original prompt prefix before the target function was retained, full fenced blocks were stripped if present, and the official tests were used. One sample per task; both variants passed 160/164 (97.56%) under this chat-adapted protocol. Not directly comparable to canonical completion-format scores.

## Completed HumanEval comparison

| Metric | Original GSQ-RCO | GSQ-RCO + AtomicChat Rank-1 |
|---|---:|---:|
| Functional pass@1 (local chat protocol) | 160/164 (97.56%) | 160/164 (97.56%) |
| Mean per-request decode | 303.61 tok/s | 288.30 tok/s |
| Total generated tokens | 150002 | 149921 |
| Normal completion | 164/164 | 164/164 |

Common failures: 47, 99, 145; original-only failure: 132; adapter-only failure: 32. The two variants share 159 passing tasks. Decode overhead is 5.04% in this run. This does not establish parity across other programming tasks or general behavior. Dataset gzip SHA256: b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef. No public model upload yet.

## Reproduction

Fetch the original adapter without duplicating weights in Git: `python -m tools.convert.fetch_inkling2_rank1 --out-dir atomicchat-adapter`. Supply the verified GGUF as `--source rank1=PATH` to the Inkling2 GGUF converter. Runtime implementation and tests are in the experimental Git branch. Local Windows benchmark helper scripts contain machine-specific paths and are not portable installers.
