# RTX 4090: Huihui IQ3S Vision DFlash2 qualification

## Status

Qualification completed, 2026-10-30. Two isolated CUDA candidates were implemented,
built and tested, then rejected and removed. No new runtime variant meets the
requested reproducible >=3% end-to-end gain without relevant regressions.
The baseline mathematics and production defaults are retained; no VRAM reduction
is established. Useful benchmark, oracle and documentation changes remain.

The target GGUF products dominate: 91.90% of plain GPU kernel time and 78.09%
with DFlash2-5. IQ3_S alone contributes 30.22% /26.33%. Metadata sharing fails
to improve whole inference materially; the T6 Tensor Core dispatch is 6–8%
slower and introduces additional greedy divergence. Draft7 and chunk4096 have
favorable cases but significant regressions elsewhere. Draft5/chunk1024 remain
the general recommendation. Existing N-Gram copy proposals are not a new speedup.

Branch: `perf/rtx4090-iq3s-dflash2`, based on fork PR #7,
`572d6859670a1d9f30a1c2e9b87ad26d7f9d43c3`. The local master initially
pointed to upstream `33163f05`; only the work branch was advanced to
`origin/master`. Both remotes were fetched. PR #7 includes that upstream
revision and retains the opt-in fork experiments and regression coverage.
Qualification was performed without changing master. Commits are made only
on explicit user request.

Continuation note: between agent turns, the work was externally committed as
`93d5dbfc` on master and upstream was merged as `ce2ff852`. The work branch had
also been moved back to `33163f05`. On continuation only the work branch was
fast-forwarded to the existing master, preserving the unrelated local startup
script. The agent did not create those commits or modify master. Initial
baseline, metadata and draft-width measurements use staged, hashed pre-merge
binaries. The MMQ/parameter comparisons and final regressions use separately
hashed current-merge builds. Evidence is not relabelled between them. The merge's
active arithmetic changes concern tensor parallelism; fresh single-GPU serving
qualification has passed on the final current-source build, below.

The seven main unprofiled campaigns contain420 measured case repetitions
(warmups, profiler captures and final smokes excluded), checked against their
per-process manifests. This is a coverage count, not420 independent repetitions
of any individual comparison; paired sample counts are given in each section.

## Artifact and host

The only inference artifact selected is the existing, read-only
`C:/ninfer-all/models/Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer`:
15,017,456,896 bytes, artifact ID `59ac51f9c2774978b3688f36043c56f8`.
No weights were downloaded, converted or edited. Directory inspection uses
the existing Python 3.11 artifact reader, not model inference.

Text is `Qwen3_5ForCausalLM`, H=5120, 64 layers, intermediate=17408,
24 query heads, 4 KV heads, head dimension 256. Its declared position limit
is 262144. DFlash2 contains five local-attention layers, window 2048,
32 query heads, 8 KV heads, head dimension 128. Its runtime draft width is
not a stored weight dimension; counts 3, 5 and 7 are supported by the
configuration contract. All requested cases through 100K have executed at
context176128; this does not test generation all the way to that capacity.
Vision and MTP are present; MTP is outside this campaign.

Unique physical objects reachable through each component's bindings, resolving
both direct and multipart bindings (object counts are not execution counts):

| Text format | Objects | Stored bytes |
|---|---:|---:|
| IQ3_S | 131 | 3642777600 |
| IQ4_XS | 89 | 3022028800 |
| IQ3_XXS | 68 | 2075279360 |
| Q4_K | 40 | 1582202880 |
| IQ2_S | 17 | 800215040 |
| Q2_K | 19 | 240844800 |
| IQ2_XS | 9 | 221265920 |
| IQ2_XXS | 6 | 98672640 |
| BF16 | 353 | 52467712 |
| IQ1_M | 1 | 19496960 |
| FP32 | 96 | 18432 |

DFlash2 binds 21 Q8 tensors (1,838,284,800 bytes) and 45 BF16 tensors
(388,508,160 bytes). Vision uses Q4/Q5/Q6/Q8 and BF16. These numbers
include physical parents once per component; they do not predict GPU time.

Host: RTX 4090, 24,564 MiB reported, WDDM with active desktop; driver 617.42,
CUDA 13.4.92, MSVC 19.44.35229, Python 3.11.9. Initial observation:
51 C, 450 W power limit, 2107 MiB total VRAM occupied. This is an idle
snapshot, not measured inference telemetry.

## Prior evidence and experiment exclusions

The complete existing [4090 workflow](../rtx4090.md),
[GGUF campaign](rtx4090-gguf-2026-10.md),
[diagnosis](rtx4090-diagnosis-2026-10.md), and
[stability investigation](rtx4090-stability-2026-10.md) govern the starting point.
Their measurements used another named artifact and are not new measurements of
this artifact. In particular, 92%/78.3% GGUF kernel shares and 128-register
T6 occupancy are historical evidence, pending current-artifact remeasurement.

Do not repeat the rejected one-row-per-warp, activation-streaming or rolled-T6
strategies without new evidence. SFU, BF16x8 SiLU and T6-loop remain OFF.
Quantized-KV plain/verification token drift is already documented; compare
candidate to baseline within the same mode and use the existing oracles.

## Baseline preparation and checks

The baseline is a fresh Release build in `build-rtx4090-huihui-base`, CUDA
architecture 89 only, tests and benchmarks enabled. Existing experiment build
directories are not accepted as baseline: their caches included rejected
ROW1/STREAM_A flags or archived experimental binaries.

The first standard-wrapper configure started rebuilding locally available
vcpkg dependencies, including FFmpeg. It was stopped before NInfer compilation.
The replacement configure reuses `build-rtx4090/vcpkg_installed`, with manifest
installation disabled, and explicitly selects Python 3.11. Initial manual
configure attempts lacked the early toolchain initialization and failed to find
FFmpeg; a fresh CMake configure with the vcpkg toolchain corrected that.
Compiler warnings from existing xgrammar sources were observed.
The initial unbounded NInfer build reduced available host RAM from about 40 GB
to 17 GB. It was stopped and resumed with `cmake --build
build-rtx4090-huihui-base -j8`; completed objects were retained. There was no
observed out-of-memory failure. Existing Attention template warnings also appear.
The first full build exposed a pre-existing link failure in
`ninfer-multi-gpu-probe`: its direct compilation of `device.cu` omitted
`peer_preflight.cpp`, producing unresolved `read_peer_device_facts` and
`peer_preflight_refusal` symbols. `tools/CMakeLists.txt` now includes that source
in the standalone probe, consistent with core's existing source list. This is
a build prerequisite fix, not a performance candidate or arithmetic change.

The unchanged baseline Python checks passed: 56 tests across
`test_bench_matrix`, `test_tune_rtx4090`, `test_speed_of_light`,
`test_gguf_bench_reader`, `test_target_trace`, and `test_engine_phases`.
The native Release build completed successfully after the standalone probe fix;
CLI and server `--help` execute. The CLI is 748526080 bytes, server 750589440
bytes and calibration executable 708041728 bytes. Binary SHA-256 records are
retained in `baseline-hashes.json`. Later sections give the completed comparisons.

The initial broad 362-test CTest attempt exposed a pre-existing Windows failure
in `ninfer_multi_gpu_test`: its Linux sysfs fixture creates PCI directory names
containing colons, which Windows rejects. The next expert-delivery test did not
complete within 180 seconds and was stopped, together with the broad run. It is
not a passing result or an established diagnosis of that asynchronous delivery
problem. That dense-model-unrelated investigation is outside this optimization.
The initial partial log and explicit stop reason are retained. A selected
215-test run covering all registered Op tests plus Graph, speculation and
protocol/report checks completed serially: 213 passed, 2 skipped, no failures,
3204.20 seconds. Skips: the real-artifact n-gram graph planning prerequisite and
the hardware-specific NVFP4 TMA staging race test. A 600-second default per-test
timeout was used (individual registered timeout properties still apply).
No existing test was weakened or disabled, and no numerical tolerance was changed.

The original `ninfer_bench` had no FP16 GDN-state option and used the Engine's
FP32 default. The measurement interface now accepts `--gdn-state-fp16`, forwards
it to the existing Engine option, and reports the actual selected option in
table/JSON/CSV. The benchmark's default remains FP32. CLI/default and JSON
contract checks were added; the rebuilt benchmark and contract test pass. This changes
neither CUDA arithmetic nor Engine defaults. Performance comparisons use this
same measurement-only change in both arms.

Local build logs, artifact directory inspection and component inventory are
under `profiles/bench/huihui-4090/`. No old binary's timing is substituted for
the new baseline.

The tiled default benchmark corpus is unsuitable as sole evidence of daily-use
speculative acceptance. A task-local 180000-token corpus was prepared from 23
ordered tracked repository documents/code files without tiling, using the
artifact's own tokenizer resource. It has 10250 distinct token IDs. The
`tokenizers` package was installed only into the ignored task output directory;
global Python packages and the model were not changed. The source manifest and
IDs are retained there. Real chat/code tool flows were subsequently exercised
with Qwen Code and the server. Exact matrix output counts require `-pg P,127` or
`-pg P,255`: the benchmark's G counts decode intervals after the first output.

## Current-artifact functional and profiling evidence

The rebuilt benchmark contract passes. The artifact starts with context/KV
capacity 176128, rk4v4, FP16 GDN state, chunk1024 and CUDA Graphs. With DFlash2-5,
the Engine reports 14360105472 weight bytes and 4179599872 runtime-reservation
bytes, including 3156213760 KV-payload bytes. These are Engine reservations,
not sampled total-board VRAM. The first complete plain matrix reaches 100000
prompt tokens and produces all 128 requested outputs without failure. Its
sampled PID peak is 14968.26 MiB; board peak 16969 MiB including the desktop.
This single process is not a distribution or an optimization result.

The existing real DFlash2 test passes with this exact artifact:
`ninfer_tests ninfer_qwen3_5_dflash2_real_test 5 1 1 1 int8 1 1`, with
`NINFER_TEST_ARTIFACT` set to the absolute model path above. It exercises Graph
acceptance/rollback, retained-prefix restore and its synthetic image/video
Vision fixtures. It reports 18/20 accepted drafts on its first request.
That integration test uses INT8 KV and the test's default FP32 GDN state;
it does not replace the requested rk4v4/FP16 performance/greedy comparisons.

Systems2026.3.2 node traces capture one measured pp2048+tg255 request after
warmup, with the custom non-tiled corpus, max context176128, rk4v4, FP16 state,
chunk1024 and Graphs. DFlash2 uses five drafts and n-gram drafts0. The existing
phase analyzer supplies decode intervals; kernel durations are clipped at every
interval boundary. An initial containment-only aggregation omitted boundary
overlaps; correcting it changes DFlash2's provisional 78.67%/26.52% figures to
78.09%/26.33%. The corrected numbers below are authoritative.

| Decode capture | GGUF vector share of summed kernel time | IQ3_S share |
|---|---:|---:|
| Plain | 91.90% | 30.22% |
| DFlash2-5 | 78.09% | 26.33% |

Host decode phase wall time outside traced CUDA activity totals about 260 ms
plain and 70 ms DFlash2 in these captures. These are elapsed intervals, not a
CPU-busy or WDDM-causality measurement. CPU scheduling/WDDM coverage and aligned
profile telemetry are incomplete. Profiled throughput is not pooled with any
unprofiled result.

Compute2026.3.1 full kernel-replay captures, first measured unfused IQ3_S/K5120
product (the 10240-row projection, 22528000 stored bytes):

| Metric | T1 | T6 |
|---|---:|---:|
| Duration us | 28.448 | 31.680 |
| DRAM read bytes | 22898304 | 23337984 |
| DRAM read GB/s | 804.92 | 736.68 |
| Registers/thread | 56 | 128 |
| Achieved occupancy | 65.95% | 30.05% |
| Global-load L1 sectors (32 bytes each) | 5500928 | 14163968 |
| L2 sectors | 915923 | 1069679 |
| Spilling requests | 0 | 0 |

Both captures use45 replay passes and warn about system-memory backup. They
support a near-one-pass DRAM stream and substantial cache traffic, not natural
cache residency for every layer or an Engine speedup. Current DRAM traffic
does not support a hypothesis of sixfold weight rereads.

The old diagnostic target-trace patch fails `git apply --check` (malformed blank
context); a task-local normalized/recount attempt also fails current-source
context matching. Neither was applied. Fresh raw-logit tracing is not yet
available from that patch. Native exact argmax, acceptance and independent
numerical checks above pass; the patch failure is not an inference failure.

## Rejected metadata candidate

The rejected experiment used `NINFER_ADA_GGUF_IQ3S_WARP_META`, initially OFF and
restricted to a single SM89 build. Only unfused IQ3_S/K5120/T1 or T6 dense products
used the alternate decoder. Eight adjacent lanes shared the block's half scale and four scale bytes
through two shuffles. No stored weights, q8_1 quantization, integer dots,
accumulation order, output conversion, launch topology or workspace change.
This targets redundant logical metadata requests, not DRAM weight rereads.
Additional shuffle/predicate and register costs are explicitly unqualified.
The candidate was compiled and tested after the unprofiled B/B campaign.

The existing greedy-comparison controller now optionally forwards
`--gdn-state-fp16` to both CLI arms, so the quality comparison can use the same
state representation as the requested performance setup. Python3.11 compilation
passes. All six baseline/candidate comparisons pass: 68-token CUDA/German and
3585-token long prompts, 64 generated IDs plus exact UTF-8 text, plain and
DFlash2-5, rk4v4, FP16 state and Graphs. That controller invocation leaves
prefill chunk automatic in both arms; it is not an explicit1024 qualification.
Cross-backend identity is not promised:
the German prompt first differs at token32, identically in both arms.

The candidate build passes GGUF Linear (including the IQ3_S FP64 oracle), CPU
GGUF and benchmark contract tests (27.81 seconds). Its real-artifact Graph,
prefix and Vision integration test also passes with the same arguments and
18/20 initial accepted drafts as baseline.

Single-kernel full-replay observations, same filter/method as baseline above:

| IQ3_S/K5120 unfused | Baseline T1 | Metadata T1 | Baseline T6 | Metadata T6 |
|---|---:|---:|---:|---:|
| Duration us | 28.448 | 27.200 | 31.680 | 29.280 |
| Registers/thread | 56 | 64 | 128 | 128 |
| Achieved occupancy % | 65.95 | 58.19 | 30.05 | 31.27 |
| DRAM read bytes | 22898304 | 22672256 | 23337984 | 22649344 |
| L1 global-load sectors | 5500928 | 5684736 | 14163968 | 14355968 |
| Spilling requests | 0 | 0 | 0 | 0 |

The T1 register/occupancy regression and increased L1 sectors are real costs.
Faster isolated captures do not prove reproducibility or an Engine improvement;
cache/replay and clock differences are not eliminated. No gain is adopted.
Six alternating A/B pairs over all five prompt lengths and both modes completed,
unprofiled, with identical fixed corpus/configuration and staged binaries. There
was no concurrent compilation or other task GPU work. The candidate failed the
3% End-to-End acceptance criterion and was removed on the work branch. Its patch
and binaries remain under the ignored task evidence directory for review.

| Mode / prompt | Baseline decode tok/s | Metadata decode tok/s | Paired median change [min,max], % |
|---|---:|---:|---:|
| Plain / 512 | 68.693 | 68.596 | -0.109 [-3.923,+1.087] |
| Plain / 2048 | 68.147 | 68.398 | +0.110 [-3.801,+8.426] |
| Plain / 8192 | 67.586 | 67.520 | -0.061 [-3.899,+0.734] |
| Plain / 32768 | 64.792 | 64.620 | -0.154 [-3.285,+0.565] |
| Plain / 100000 | 59.685 | 59.381 | +0.016 [-1.110,+0.284] |
| DFlash2-5 / 512 | 165.139 | 165.189 | +0.047 [-0.206,+0.459] |
| DFlash2-5 / 2048 | 175.647 | 175.886 | +0.176 [-0.676,+0.390] |
| DFlash2-5 / 8192 | 173.714 | 173.932 | +0.091 [-0.084,+0.203] |
| DFlash2-5 / 32768 | 168.816 | 168.835 | +0.052 [-0.161,+0.287] |
| DFlash2-5 / 100000 | 109.461 | 109.582 | +0.100 [-0.092,+0.212] |

Rates are arm medians; changes are medians of paired ratios, **not** ratios of
the arm medians. Output counts are128 at 512/100000 and 256 otherwise. Plain
prefill paired medians span -0.641% to+0.134%, individual pairs -4.598% to+5.759%;
TTFT medians -0.135% to+0.647%, individual pairs -5.445% to+4.836% (positive
TTFT change is worse). DFlash2 prefill medians span -0.065% to+0.226%, individual
pairs -1.683% to+2.552%; TTFT medians -0.220% to+0.065%, individual pairs
-2.489% to+1.714%. Request-total-time medians span -0.134% to+0.256% plain and
-0.074% to-0.022% DFlash2. These small mixed observations establish no useful
latency or throughput improvement; adverse samples are retained.

Sampled PID-dedicated peaks are identical between arms:14968.26 MiB plain and
17610.28 MiB DFlash2. Engine weight/runtime reservations also match exactly:
11755300864/3567728128 bytes plain,14360105472/4179599872 bytes DFlash2.
No VRAM saving was observed. Across arms, board peaks span16451–16676 MiB
plain and 19075–19093 MiB DFlash2, including the changing desktop. Recorded core
clocks span2415–2685 MHz, memory10501 MHz, temperatures38–78 C and sampled board
power peaks up to 456.60 W under the existing450 W limit. These are whole-process
samples, not phase-only maxima or proof of absent sub-sample interference.

Benchmark SHA-256:

- Baseline: `60e76886aa3fd659e40b87caeda2ebf522075160acee39069848d942e7699035`
- Metadata: `503808ed18168f2cbcf4efc9a6464b07b5debcd652e2facc0ca751ab7430c824`

Full paired summaries, every individual sample and resource telemetry summaries:
`profiles/bench/huihui-4090/meta-ab-full/{summary.json,samples.csv,resources.json}`.
The rejected source diff is `profiles/bench/huihui-4090/meta-candidate.patch`.

## Completed baseline variance series

Three identical-binary B/B pairs per mode, AB/BA order, one discarded warmup
and one measured repetition per case/process, all five requested prompt/output
counts, max context176128, rk4v4, FP16 state, chunk1024, Graphs, concurrency1.
Clocks are unlocked, existing power limit450 W. Memory clock remains10501 MHz
in the inspected busy samples; core clocks and temperature vary. This is a
daily-use WDDM/desktop condition, not a fixed-clock experiment. Raw telemetry
includes startup, warmups and idle periods; sampled peaks are not instantaneous
allocator maxima. Each process records the exact executable SHA-256.

Paired Decode B/A changes, percent; favorable and adverse observations retained:

| Prompt/output | Plain median [min,max] | DFlash2-5 median [min,max] |
|---|---:|---:|
| 512/128 | +0.09 [-8.63,+0.62] | +0.04 [-13.03,+3.89] |
| 2048/256 | -0.46 [-1.30,+0.05] | -0.23 [-1.45,+0.70] |
| 8192/256 | +0.13 [+0.07,+2.62] | +0.56 [-5.75,+4.82] |
| 32768/256 | +0.04 [-0.08,+1.41] | +0.07 [-0.87,+10.85] |
| 100000/128 | +0.33 [-0.46,+0.66] | -1.37 [-2.46,+0.11] |

Every same-case DFlash2 B/B run has matching round/draft/accept counts. The
short-case outliers and 32K positive excursion are therefore not explained by
a changed speculative work count. No causal diagnosis is claimed. Plain
prefill's first pair spans -9.57% to+8.86%; the raw reports retain TTFT too.
Complete paired summaries and every individual sample are under
`profiles/bench/huihui-4090/bb-full/{summary.json,samples.csv}`.

The 2K DFlash2 Systems capture also separates GPU kernels by the native
prepare-verify and speculative-sampling boundaries: 16.98 ms/round target
forward/argmax, 3.015 ms draft plus context materialization, 0.145 ms GPU
transition/acceptance. The Engine's complete profiled decode round averages
21.08 ms. There are73 rounds,364 drafted and 182 accepted tokens: 2.493 accepted
drafts/round,3.493 outputs/round including the verified token. Component GPU
rates are352.58 valid target columns/s and 1654.04 draft tokens/s including
context materialization. These denominators differ from output throughput and
are not pooled with unprofiled measurements. Boundary/host elapsed time is not
CPU-busy time. Target arithmetic remains the dominant measured component.

## Rejected T6 Tensor Core dispatch

A second, distinct mechanism tried the existing IQ3_S MMQ Tensor Core route
at T6, normally handled by the vector path. The opt-in SM89-only flag selected
IQ3_S rows>=5120 divisible by128, K5120/17408; other shapes/formats retained
their existing route. Mixed-product activations were kept separately for the
matrix/vector layouts, with explicit workspace-query coverage and unchanged
stored weights, epilogues and precision tolerances. No runtime repacking.
This was implemented alone after removing metadata sharing.

Fresh current-merge binaries, both with metadata/SFU/BF16x8/T6-loop off:

- Baseline SHA-256: `a8630a41a34376fe51373513ea9a12ec722b75f094137a7debab990cfdaf1758`.
- MMQ SHA-256: `4b0e5c55edd0d14c0d299985ed8abc643f3db8ecfd3b170978e5efa493e7bcec`.

Four selected CTests pass (GGUF independent oracles, MMQ store, Graph and
benchmark contract),29.15 seconds. Additional real shapes include
K17408/N5120/T5,6,7 and K5120/N5119,5120,5121/T6. CUDA13.4 Compute Sanitizer
memcheck reports 0 errors for the focused IQ3_S verification cases. The actual
artifact DFlash2 K5/Graphs/Vision/reuse regression passes,18/20 accepted,
state transfers4 D2H/2 H2D (that fixture uses INT8 KV, not the matrix's rk4v4).

Six alternating unprofiled A/B pairs at 2K and 32K decisively reject the route:

| Prompt/output | Baseline / MMQ decode tok/s | Paired median [min,max], % |
|---|---:|---:|
| 2048/256 | 171.535 /157.331 | -8.237 [-8.747,-7.512] |
| 32768/256 | 164.904 /155.004 | -6.020 [-6.467,-5.892] |

All12 paired case observations are slower. First-pair round times increase
20.328→23.568 ms at 2K and 21.747→24.518 ms at 32K despite fewer rounds:
73→69 and 71→67. Thus an acceptance increase does not rescue the slower work.
Prefill paired medians+0.591%/-0.167%, TTFT-0.587%/+0.168%; no broad latency
benefit is established. PID memory peak rises17610.28→17622.28 MiB despite
identical weight/runtime reservation ledgers. Board peaks19288→19299 MiB,
temperature42–77 C, core210–2685 MHz including startup, memory810/10501 MHz,
sampled power up to 454.93 W under450 W. Full data and total-time distributions:
`mmq-ab-screen/{summary.json,samples.csv,resources.json}`.

Greedy comparisons at explicit chunk1024/rk4v4/FP16 state pass all three plain
prompts and one of three DFlash2 prompts. German and long DFlash2 outputs
**differ from their same-mode baseline**; this is additional candidate divergence,
not merely the known baseline plain-versus-DFlash2 difference. No quality-parity
claim is made and no tolerance was relaxed. The route and its CMake option were
removed. The rejected patch (also showing the prior metadata removal) and staged
binaries are retained; tests retain useful independent shape coverage.
The other matrix cases and six Plain A/B pairs were not run for this rejected
candidate; the 2K/32K regression already settles its disposition.

## Other measured bottlenecks and draft widths

After the metadata candidate failed, the next measured costs were checked from
the same phase-clipped trace rather than continuing metadata variants: IQ4_XS
accounts for 23.60% plain /18.49% DFlash2 GPU kernel time, IQ3_XXS17.21% /14.34%,
Q4_K12.40% /11.33%. The vector Q8_1 activation-quantization kernel is only
2.12% /1.74%; even eliminating it cannot alone provide a3% end-to-end gain.
The largest individual non-GGUF DFlash2 kernel is the draft's Q8 fused FFN
gate/up product at 5.19%, followed by Q8 down at 2.74%. This prioritizes target
weight products over speculative transition changes or another activation-only
experiment. Full groups and non-vector kernels are in `format-shares.json`.

Six alternating A/B pairs for each comparison, all five requested cases, same
hashed baseline executable and corpus, rk4v4/FP16 state/Graphs/chunk1024,
concurrency1/context176128. Neither experiment has n-gram drafting enabled.
These are independent campaigns; their K5 medians differ from the earlier
metadata campaign and must not be substituted into that comparison.

| Prompt/output | K5 / K3 decode tok/s | K3 paired change [min,max], % | K5 / K7 decode tok/s | K7 paired change [min,max], % |
|---|---:|---:|---:|---:|
| 512/128 | 161.141 / 146.857 | -8.834 [-9.138,-8.565] | 161.092 / 161.643 | +0.293 [+0.271,+0.548] |
| 2048/256 | 171.473 / 158.215 | -7.741 [-9.076,-7.550] | 171.280 / 163.078 | -4.751 [-4.982,-4.536] |
| 8192/256 | 169.343 / 166.377 | -1.788 [-1.997,-1.525] | 169.363 / 146.615 | -13.410 [-13.888,-13.186] |
| 32768/256 | 164.718 / 162.861 | -1.239 [-1.574,-0.870] | 164.540 / 178.886 | +8.689 [+8.558,+9.336] |
| 100000/128 | 106.915 / 102.377 | -4.278 [-5.329,-4.163] | 106.919 / 87.738 | -17.897 [-18.007,-17.186] |

Higher acceptance percentage is not itself useful: at 2K, K3 accepts66.93%
versus K5's50.00%, but only 2.000 versus 2.493 draft tokens/round. A faster
18.962 versus 20.371 ms/round does not compensate for 85 versus 73 rounds.
K7 at 2K accepts37.78%,2.592 drafts/round,22.024 ms/round,71 rounds.
At32K K7 accepts3.233 drafts/round versus K5's2.577 and needs60 versus 71
rounds; at 100K this reverses to 1.250 versus 1.540 and 56 versus 50 rounds.
These workload-dependent results do not justify a global K7 default or a
context-only adaptive policy from this single corpus. K5 remains the general
recommendation; the 32K gain is a measured case-specific option with significant
regressions elsewhere. Its+8.689% paired median also does not exceed the earlier
32K identical-binary positive excursion of+10.854%; it is not a qualified win
under the requested above-baseline-variation criterion.

Prefill paired medians for K3 versus K5 span-0.013% to+0.826%, TTFT-0.829%
to+0.013%; K7 prefill-0.902% to+0.056%, TTFT-0.056% to+0.891%.
Full per-case distributions, total latency and all samples are retained in
`draft5-vs{3,7}-full/{summary.json,samples.csv}` under the task evidence directory.
Sampled dedicated PID peaks are17606.28/17610.28/17616.28 MiB for K3/5/7;
runtime reservations4174907392/4179599872/4184292352 bytes. K3 saves only 4 MiB
of sampled device memory while slowing every case. Board peaks reach19387 MiB
including desktop use. Whole-process telemetry spans210–2700 MHz core,
810/10501 MHz memory (startup/idle included),37–78 C, up to 454.01 W sampled
under450 W limit. Busy K3/K7 arm memory samples are10501 MHz. Details and
acceptance/round measurements are in `draft-comparison-details.json`.

Separate Systems profiles at 2K use the same original baseline weights/config
and phase-clipping method as K5. They are not throughput samples:

| Draft count | Target forward/argmax GPU ms/round | Draft+context GPU ms/round | Transition GPU ms/round | Engine ms/round | Target valid columns/GPU s | Draft proposals/draft-context GPU s |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 15.748 | 3.048 | 0.136 | 19.924 | 253.25 | 980.27 |
| 5 | 16.979 | 3.015 | 0.145 | 21.084 | 352.58 | 1654.04 |
| 7 | 18.597 | 3.045 | 0.148 | 22.804 | 422.60 | 2252.97 |

The rate denominators differ; none is end-to-end output throughput. Target
arithmetic dominates all three widths. GPU transition kernels are small;
host boundary wall time is not certified CPU busy time. Wider target throughput
does not guarantee better accepted-output throughput. Raw profiles and
`profile-dflash{3,7}-gpu-parts.json` retain the separation.

The next-largest format IQ4_XS was then profiled at the actual T6/K5120 route:
33.664 us,28.026496 MB DRAM read (~832.54 GB/s),134 registers/thread,24.04%
achieved occupancy,12,800,000 L1 global-load sectors and 0 spilling requests.
This is one 45-pass kernel-replay capture, with system-memory backup warning;
it does not establish every layer's behavior. It supports investigating
shape-specific schedules/register allocation, not another unmeasured metadata
sharing variant. Near-streaming DRAM bandwidth limits the plausible gain.
Evidence: `ncu-iq4-t6-baseline.{ncu-repz,csv,log,json}`.

## Prefill chunk 1024 versus 4096

After rejecting the two CUDA candidates, an existing scheduling parameter was
evaluated separately. Six alternating full-matrix pairs use the same current
baseline executable/hash in both arms, DFlash2-5/ngram0/rk4v4/FP16 state/Graphs,
context176128, concurrency1; only prefill chunk differs. No build runs alongside
these unprofiled measurements.

| Prompt/output | Decode tok/s chunk1024 /4096 | Paired median [min,max], % |
|---|---:|---:|
| 512/128 | 161.125 /161.439 | +0.157 [-0.254,+0.500] |
| 2048/256 | 171.149 /195.129 | +13.964 [+12.753,+14.182] |
| 8192/256 | 169.522 /154.465 | -8.921 [-9.021,-8.641] |
| 32768/256 | 164.525 /171.492 | +4.306 [+3.476,+5.238] |
| 100000/128 | 106.874 /93.475 | -12.545 [-12.631,-12.420] |

| Prompt | Prefill tok/s chunk1024 /4096 | Engine TTFT seconds1024 /4096 | Total seconds1024 /4096 | Paired total-time median [min,max], % |
|---|---:|---:|---:|---:|
| 512 | 3583.06 /3575.41 | 0.1430 /0.1433 | 0.9316 /0.9302 | -0.111 [-0.249,+0.250] |
| 2048 | 3633.13 /3825.56 | 0.5638 /0.5355 | 2.0541 /1.8429 | -10.284 [-10.391,-7.911] |
| 8192 | 3596.65 /3722.12 | 2.2779 /2.2010 | 3.7832 /3.8529 | +1.980 [+1.707,+3.152] |
| 32768 | 3314.67 /3421.35 | 9.8863 /9.5778 | 11.4355 /11.0660 | -3.242 [-3.304,-3.046] |
| 100000 | 2727.12 /2816.70 | 36.6705 /35.5035 | 37.8590 /36.8632 | -2.626 [-2.698,-2.608] |

The prefill paired medians are-0.342/+5.292/+3.284/+3.234/+3.273%; TTFT
+0.338/-5.026/-3.182/-3.136/-3.171%, in table order. One2K pair has worse
prefill-1.119% and TTFT+1.129%; short-case total time has a+0.250% adverse
pair. All8K pairs have worse total latency despite improved prefill. Favorable
results are therefore not a general gain without regressions.

The large decode changes mainly track changed speculative work, not a faster
decode kernel: first-pair2K rounds 73→64 with 20.478→20.457 ms/round;100K
rounds 50→57 with 23.815→23.856 ms/round. Changing supported prefill schedules
changes finite-precision state and downstream proposals. The long3585-token
greedy fixture differs in both modes; two short fixtures match in both modes.
These are64-token generated-prefix comparisons (including model reasoning),
not completed-answer accuracy or a proof of quality loss/parity.

Chunk4096 increases sampled PID peak17610.28→18356.29 MiB, about 746 MiB,
and runtime reservation4179599872→4961805824 bytes; weights are identical.
Board peaks19180→19926 MiB including desktop, core2415–2685 MHz, memory
10501 MHz,47–78 C, sampled power up to 451.12 W under450 W. This is a cost,
not a VRAM optimization. The general recommendation remains1024. No Plain
six-pair timing series was run for this rejected parameter; its behavior is
not described as unchanged. All reports, paired ranges and resources:
`chunk1024-vs4096/{summary.json,samples.csv,resources.json}` and
`chunk4096-quality/{summary.json,backend-comparison.json}`.

## Existing N-Gram copy route

The isolated DFlash2 baseline explicitly disables copy proposals. Normal serving
defaults to 15 when speculation is selected, and the user's existing launcher
also selects15. A six-pair screen therefore compares0 versus 15 at 2K/8K/32K,
same current baseline binary, chunk1024 and all other controlled settings.
This is an ablation of an existing capability, not a new kernel or a gain over
the user's existing default.

| Prompt/output | Decode tok/s ngram0 /15 | Paired median [min,max], % | Total-time paired median [min,max], % |
|---|---:|---:|---:|
| 2048/256 | 171.365 /170.565 | -0.475 [-0.792,-0.345] | +0.579 [+0.456,+0.628] |
| 8192/256 | 169.461 /170.956 | +0.891 [+0.756,+1.598] | -0.347 [-0.735,-0.165] |
| 32768/256 | 164.756 /164.429 | -0.159 [-0.368,-0.081] | +0.009 [-0.088,+0.167] |

At2K the first ngram15 sample uses one copy round,9 proposed/3 accepted tokens;
8K uses one,15 proposed/0 accepted;32K uses none. This untiled corpus does not
establish the value of copy proposals for strongly copying tasks. The small
mixed effects fail the 3% criterion. Prefill paired medians-0.044/+0.137/+0.064%,
TTFT+0.702/+0.025/-0.016%; all 2K TTFT pairs worsen (+0.164..+1.059%).
PID peak17610.28→17708.28 MiB (+98 MiB); runtime reservation grows
4179599872→4706378752 bytes, weights unchanged. Board peaks19180→19278 MiB,
core210–2685 MHz including startup, memory810/10501 MHz,48–78 C, power up to
452.52 W under450 W. No VRAM saving or general gain was found.512/100K and
six Plain timing pairs were not run for this rejected optimization hypothesis.
Data: `ngram0-vs15-screen/{summary.json,samples.csv,resources.json}`.

## Final source and numerical regression qualification

The final Release build uses only SM89/CUDA13.4; metadata sharing and the T6
MMQ experiment/option are removed. Existing SFU sigmoid/SiLU, SFU softplus,
BF16x8-SiLU and T6-loop options remain OFF, as does BF16 residual addition.
No production route, stored quantization, API semantics or accuracy criterion
was replaced with an experimental implementation.

All15 affected native CTests pass on the final binary,303.20 seconds: Graph,
benchmark report/options, argmax, DFlash2 attention, speculative round,
GGUF independent oracles/MMQ store, N-Gram and wide-accept/distribution tests,
and CPU/GPU GGUF expert equality. The actual artifact's K5/Graph/Vision/reuse
test passes again (18/20 accepted,4 D2H/2 H2D). All six exact greedy
comparisons against the current-merge baseline pass at explicit chunk1024,
rk4v4, FP16 state and Graphs,64 output tokens each. This establishes the
tested generated-prefix stability after removing the experiments; it is not
a general task-quality score. The earlier213-pass/2-skip baseline and initial
broad-suite Windows failure/stall remain disclosed above; no all-suite pass
is claimed.

Final SHA-256 records in `final-hashes.json`:

- Benchmark: `0db759e42121ce978437023884de70ca533a8b135d4718c66db2e0a5166e5f42`.
- CLI: `b53147c5892887902288fce8a4a51b10a6fa17de065bd090cab50bd0a52618c3`.
- Server: `bbaa4ffd244337bc90328e26ec48a42a95d6adb89bb74b081ebd1eb27ba6df1f`.
- Tests: `e918d23c74d792b8436ce01355ef6c899b33f1ca516dda00a21d19148b158e91`.

Logs: `build-final.log`, `final-ctest.log`, `final-real.log`,
`final-quality/{summary.json,backend-comparison.json}`. Python3.11 compilation
of the changed quality controller and `git diff --check` pass.

Retained task changes include the FP16-state benchmark interface and report
tests, the standalone Windows peer-probe build dependency fix, extended IQ3_S
oracle shapes/focused sanitizer selection, explicit per-arm prefill chunks in
the existing greedy controller, removal of the rejected metadata option/kernel
route, and linked documentation. `gguf_linear.cpp`'s experimental MMQ dispatch
was restored completely; no unqualified performance refactor remains there.
Some earlier task changes were externally committed between turns, as described
in the lineage note; this list is not limited to the final uncommitted diff.

The final benchmark's embedded CUBIN inventory contains only `sm_89`, verified
with CUDA13.4 `cuobjdump --list-elf` (`final-cubins.txt`). Both final smoke
matrices complete512/128,2048/256,8192/256,32768/256 and 100000/128 with exactly
the requested outputs, rk4v4/FP16 state/Graphs/chunk1024/ngram0/context176128.
These one-repetition, zero-warmup checks certify execution, not an optimization;
their timings are not pooled with the one-warmup A/B data.

The inference matrices use prepared token IDs through the public Engine and
measure Engine TTFT/total time, not HTTP or Qwen Code end-to-end latency. Server
and client checks are functional evidence. No unmeasured server speedup, complete
answer-quality equivalence, or full176128-token-generation result is claimed.

## Current-source server and Qwen Code qualification

The existing `build-rtx4090-coding/apps/ninfer-serve.exe` reports merge
`ce2ff852`; its Release/SM89/CUDA13.4 cache has all earlier experiments off.
SHA-256 `a781473627747d86740631e9f4dc085a364031a88cd9f63f48c5650b00d34b30`.
It was exercised on 127.0.0.1:18089, alias `huihui-local`, context/KV176128,
rk4v4/FP16 GDN/Graphs, concurrency1, chunk1024, DFlash2-5, n-gram0,
resident Vision, structured-output enforcement and host-cache budget1024 MiB.
This was functional testing during CPU compilation, not throughput evidence.
Startup reported13.6 GiB weights,4.64 GiB runtime and 4.13 GiB free; the smaller
host budget was for this test, not a validated daily-use cache optimum.

Passed: models listing, greedy Chat Completions, automatic and forced
`read_file` calls with valid arguments and `tool_calls` finish reason, both
tool-result round trips, SSE content/usage/`[DONE]`, repeated-prefix reuse
(3761 cached tokens), and existing `examples/cli/media/visual_chart.png`
through the public image-url API (title number731 correctly returned).
The initial launch incorrectly used benchmark CLI names `--weights`/`--max-ctx`;
it failed before load, was corrected to positional artifact/`--max-context`,
and the failed attempt's logs were retained. No product regression was involved.

The installed Qwen Code client identified itself as0.26.4 at execution, with
explicit local OpenAI base URL/model/auth, `--bare --safe-mode`, no telemetry
or chat recording, bounded turns/tool calls/wall time and a fixture-only prompt.
Safe mode warns that `--core-tools` is ignored; the actual transcript confirms
exactly one successful `read_file`, no other executed tools, no permission
denials, zero added/removed lines, and the exact unseen fixture nonce returned
after the tool result. Two requests completed with no API errors and 4176
cached input tokens on the follow-up. The script's unused relative-path lookup
failed in the fixture working directory; it did not affect client execution or
the verified tool result. Neither client nor server was left running.
Evidence: `server-current2/{manifest.json,qa/results.json,qwen-code.log,requests.jsonl}`
under the task directory; first failed startup in `server-current/`.

The **final** server binary above was then tested again with the same model,
rk4v4/FP16 state/Graphs/chunk1024/concurrency1/context176128, resident Vision,
structured output, normal ngram15 and host-cache budget8192 MiB. Startup reported
13.6 GiB weights,5.13 GiB runtime and 4.04 GiB free. All direct API checks pass
again, including 3761 cached tokens and Vision number731. Qwen Code0.26.4 again
executes exactly one successful `read_file`, returns the exact fixture nonce,
reports zero added/removed lines and no API errors; its follow-up reuses4238
cached tokens. The unused relative lookup and ignored core-tools flag from the
earlier harness invocation are absent from this final invocation. The final
server was stopped after verification. Evidence: `server-final/`.

## Recommended local launch and remaining limits

Use the final default-math binary. The following mirrors the final functional
configuration (port18089 was used for tests;8080 below is the normal local port):

```powershell
.\build-rtx4090-huihui-base\apps\ninfer-serve.exe `
  "C:\ninfer-all\models\Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer" `
  --model-id huihui-local --host 127.0.0.1 --port 8080 `
  --max-context 176128 --kv-capacity 176128 --max-concurrency 1 `
  --kv-dtype rk4v4 --gdn-state-fp16 --prefill-chunk 1024 `
  --spec dflash2 --draft-tokens 5 --ngram-draft-tokens 15 `
  --vision --structured-output --host-cache-mib 8192
```

Point Qwen Code's OpenAI base URL to `http://127.0.0.1:8080/v1`, model
`huihui-local`. The controlled full-matrix performance figures use ngram0;
set`--ngram-draft-tokens 0` to reproduce them. Ngram15 retains the existing
daily-use copy capability and passed final serving tests, but has only the
2K/8K/32K timing screen here and no established general throughput advantage.
The user's existing startup script was not modified. No experimental build
option should be enabled from these results.

Baseline prefill/TTFT context for the six-pair metadata campaign's A arm
(text-only, no resident Vision; do not substitute later campaign rates):

| Prompt/output | Plain /DFlash2-5 prefill tok/s | Plain /DFlash2-5 Engine TTFT seconds |
|---|---:|---:|
| 512/128 | 3627.57 /3634.12 | 0.1413 /0.1410 |
| 2048/256 | 3687.30 /3702.79 | 0.5555 /0.5532 |
| 8192/256 | 3634.47 /3673.18 | 2.2541 /2.2304 |
| 32768/256 | 3347.12 /3383.95 | 9.7902 /9.6837 |
| 100000/128 | 2744.40 /2783.52 | 36.4387 /35.9265 |

There is no qualified optimized counterpart to these baselines. The favorable
and adverse candidate results above, not a fabricated speedup column, are the
outcome. Prefix-reuse correctness was tested separately; no controlled HTTP
prefix-cache latency speedup or100K prefix-restore latency was measured.
CPU scheduling/WDDM coverage and the old raw-target-trace patch remain incomplete.
The initial broad test failure/stall and two selected-baseline skips are not
overridden by the passing final15 affected tests. Rejected experiments did not
receive every later matrix/quality check once their disposition was decisive;
their specific coverage limits are disclosed in their sections.

## Concrete next optimization questions

### 2026-11-03: IQ4_XS register-cap screen

A local, isolated probe tested the existing dense IQ4_XS vector kernel at
`K=5120, T=6`, changing only its launch bounds to require four resident
128-thread blocks. Both variants use `ggml_bridge_vec.cuh` from master
`237cdff9`; the candidate copies its kernel and adds the second launch-bound
argument, four. RTX 4090, driver 617.42, CUDA 13.4, MSVC 14.44:
registers fell from 134 to 128, computed resident blocks increased from
three to four, and neither variant spilled. No production kernel or route
was changed. The candidate is rejected.

Inputs were independently generated stored IQ4_XS blocks (random six-bit scale
codes, non-linear nibble values, fixed exactly represented block scale
1/512) and Q8 activation integers with scale 1/256. A naive FP64 dot decoded
every stored weight independently. At all ten shapes below, both kernels
passed relative RMS <1e-5 and maximum error/reference RMS <1e-4; worst
observed values were 1.23e-8 and 3.08e-7. Their FP32 outputs were bit-identical.
This checks these represented inputs, not arbitrary scales or other routes.
The oracle covers the first weight bank of each shape; the seven other
timing banks were not compared numerically.
Retained artifact metadata identifies IQ4_XS objects with K=5120 and
N=1024,6144,7168,10240,16384,17408,34816. Some objects supply bound views;
these are physical object shapes, not a complete runtime-call inventory.
N=255,257,5120 are additional synthetic controls at this K. Only the
unfused dense T6 kernel is screened.

Nine order-alternating event-timed pairs per shape, 256 launches per arm,
rotated eight weight banks. Times are CUDA event microseconds per launch,
including any host-submission gaps; paired changes
are candidate/baseline duration, so positive means slower.
The initial screen covered N=255,257,5120,17408. The other six shapes were
added after checking the artifact metadata; the original pairs are retained.

| N | Baseline median us | Candidate median us | Paired median change | Full pair range | Slower pairs |
|---|---:|---:|---:|---:|---:|
| 255 | 6.263 | 6.212 | +0.53% | -12.18 to +56.47% | 5/9 |
| 257 | 6.248 | 5.932 | -6.77% | -28.33 to +10.82% | 2/9 |
| 1024 | 7.955 | 8.448 | +11.55% | -30.86 to +32.15% | 5/9 |
| 5120 | 18.312 | 20.676 | +8.16% | +6.40 to +22.94% | 9/9 |
| 6144 | 23.868 | 24.060 | -4.05% | -19.54 to +15.84% | 4/9 |
| 7168 | 27.868 | 26.832 | -4.26% | -15.68 to +9.51% | 4/9 |
| 10240 | 38.043 | 36.763 | -2.03% | -10.36 to +7.69% | 3/9 |
| 16384 | 57.755 | 59.136 | +1.70% | -3.51 to +7.07% | 6/9 |
| 17408 | 59.008 | 62.956 | +1.19% | -1.57 to +13.65% | 7/9 |
| 34816 | 119.044 | 123.055 | +0.31% | -11.90 to +18.75% | 5/9 |

Clocks were not locked: the desktop GPU moved from 330/810 MHz before the
first screen to 2535/10501 MHz afterwards, at 42–43 C. The shape extension
moved from 360/810 to 2685/10501 MHz, at 41–57 C. CPU compilation and another
application were active; no CPU/WDDM attribution was collected. Short
shapes have substantial spread, and eight small matrices fit in L2, unlike
the larger shapes. These are bounded screening results, not stable
production latency estimates. All adverse pairs remain in the results.
Some artifact shapes have favorable paired medians, but every added shape
also has slower pairs. The regressions and spread give no reason to advance this candidate to
whole-Engine A/B or broader qualification. No end-to-end change is claimed.

Probe source, compile log, raw pairs and GPU snapshots remain locally under
`profiles/bench/rtx4090-release-20261103/iq4-probe/`.

### Remaining questions

- IQ4_XS is the next weight-format group:18.49% DFlash2 GPU kernel time,
 134 registers and 24% occupancy in the captured T6 route. Investigate schedules
 for actual matrix shapes and instruction/register lifetime, with DRAM already
 near a one-stream bandwidth limit; another IQ3 metadata/rolled-loop variant is
 not justified by this campaign. A microkernel result still needs Engine A/B.
- Verification streams target weights every round. A proposal-window policy
 based on measured accepted outputs per round divided by complete round cost
 could select between prequalified Graph widths. The opposite K7 outcomes at
32K and 100K justify that question, not a context-only rule or an untested default.
 It requires independent numerical/rollback checks and held-out chat/code tasks.
- Plain decode has measured time outside CUDA activity. Obtain complete aligned
 CPU scheduling/WDDM/queue evidence before changing host submission or wait
 policy; the existing phase wall times cannot identify CPU busy time or its
 cause. Additional activation-only work has a small measured1.74% DFlash2 share.
- Long-prefill profiling can localize the useful part of chunk4096 without its
746 MiB extra reservation and changed speculative work. Preserve the GDN/state
 accuracy contract; the observed3–5% prefill gain alone is not sufficient to
 accept the 8K/100K decode regressions.

All raw evidence is local under ignored `profiles/bench/huihui-4090/`, following
the repository's profile convention; source/report changes were prepared
on `perf/rtx4090-iq3s-dflash2`. No model file was edited; this agent did not edit master.
