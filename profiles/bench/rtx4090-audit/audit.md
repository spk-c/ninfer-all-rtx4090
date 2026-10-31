# RTX 4090 audit — 2026-10-31

Current implementation: **b6c0e4d2416542ac32e2f474077cb9049852bb47**. Two Windows
deadlocks and a reproduced selected-weight corruption race are corrected. All
real-model/cache/measurement campaigns were rerun on this build; current figures
are in “Post-fix requalification” below. Its final 372-test run completed: **340 passed, 32 skipped, 0 failed**, in **2087.68 s**.
Earlier campaigns are retained with their actual source revisions and failures.

## Scope and provenance

This audit validates PR #10 on `feature/rtx4090-upstream-stability-cache-profiling`.
Starting revision: `8b66e2536f30717cc84547424c079f261a465b63`.
The local branch already tracked `origin`; both remotes advertised the same branch name.
`origin` uniquely names `https://github.com/spk-c/ninfer-all-rtx4090.git`.
Fetched that remote and set repository-local `checkout.defaultRemote=origin` to remove
ambiguous automatic branch discovery. No checkout or worktree was necessary because the
correct branch was already active. The untracked user PowerShell file was left untouched;
its initial SHA256 is in `environment.json`.

Protected refs at entry: master `a85c597f0b96686197712c4118deb2647ba7a75c`,
release tag object `4ddc76ae318b88ef6cfe99dcb9f3750ecd8b7f42`.
Neither is a target of this work. No release or merge is authorized by the campaign.

| Audit commit | Completed change |
| --- | --- |
| `aaa45c9c5261871dd0dbaff54f8d82b01cc0a681` | Initial measurement failure/configuration fixes |
| `b04550acf9547199de8cb80828f85f01dc76e8f0` | Portable peer preflight and layout boundary qualification |
| `0a40ca018b839c78fdd8dcce8eb44d82db760e49` | Windows hybrid Graph DMA-queue deadlock |
| `746c44fc42416c4b99906d2d65d44590eb8846b6` | Windows IOCP caller-to-worker handoff deadlock |
| `493fb8dbfd6e311f12f9ee7e225e1d4256bff69c` | Concurrent Linear test on actual supported device routes |
| `1922c64a8052d79dd35466891e96021ceafb9445` | Live coding-cache tool and real DFlash2 cache scenarios |
| `7109c9f7bcdab1c885a92ad1b1084e111358882c` | Qualified measurement tools and explicit copy-draft configuration |
| `b6c0e4d2416542ac32e2f474077cb9049852bb47` | Order padding clears before weight uploads; independent CPU oracle and selected-upload stress |

Earlier graph/A/B/counter measurements use freshly rebuilt `1922c64a` executables;
the fourth complete CTest build is `7109c9f7`. The first two HTTP campaigns use
the previously rebuilt `746c44fc` server: intervening493fb8db/1922c64a changes
affect tests/tools, not that server's Engine implementation. The working real-cache
fixture additions were subsequently committed in 1922c64a. Post-padding-fix real
tests, HTTP campaigns and measurements all use freshly rebuilt b6c0e4d2 binaries.
No pre-fix binary is presented as the final corrected implementation.
There is no optimized pre-audit versus post-audit fork throughput baseline.

## Environment and workload

- Windows 10 Pro 10.0.19045, AMD Ryzen 9 9950X3D (16 cores / 32 logical CPUs),
  approximately 61.6 GiB visible physical RAM.
- NVIDIA GeForce RTX 4090, WDDM, 24,564 MiB board memory; driver 617.42.
- CUDA Toolkit 13.4, nvcc 13.4.92; native architecture 89; VS 2022 Build Tools
  17.14.42, MSVC 14.44.35207; CMake 3.29.2; Ninja 1.12.0.
- Python 3.11.9. The pre-existing CMake build selected Python 3.13; the final
  validation explicitly selects 3.11. NumPy and Jinja2 are required test dependencies.
- Requested Nsight Systems **2026.1.3.425**, found recursively at
  `C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe`.
  Nsight Compute 2026.3.1 is also installed.
- Artifact: `C:\ninfer-all\models\Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer`,
  15,017,456,896 bytes, artifact ID `59ac51f9c2774978b3688f36043c56f8`.
  The production Python reader recognizes v3, `Qwen3_5ForCausalLM`, 64 layers,
  hidden size 5120 and vocabulary 248320. Stored tensors include 131 IQ3_S,
  68 IQ3_XXS and 89 IQ4_XS objects; tensor counts are not execution-time shares.
- Target configuration: DFlash2, five draft tokens, optimized proposal head,
  rk4v4 KV, FP16 GDN state. The graph-memory sweep uses prefill chunks of 1024.

At entry the GPU used 1579 MiB with 3% reported utilization. Desktop processes,
Chrome, Thunderbird, NVIDIA overlays and other applications remained open. No
listed process was a CUDA inference server. Measurements are therefore WDDM desktop
observations, not an isolated headless GPU guarantee. No other application was killed.

## Upstream changes reviewed

The following source commits were fetched through the connected GitHub API and
compared with this fork's current implementations, rather than assumed applicable
from titles or older reports.

| Source | Decision and reason |
| --- | --- |
| Neroued `d602894abcd071c273003f021258f46608aab447` | Allocation rollback already exists in `LayoutBuilder::add`: align into a local offset, check addition, then commit cursor and peak. PR #10 additionally rejects bounds and address overflow in `LayoutRegion::bind`. Retained that fix and added address-wrap and invalid-alignment regression cases. |
| Neroued `6dd2b1b4e3db715b5f68bd4163a86f1c16875a11` | Not transplanted. It changes per-rank byte-capacity startup planning, including explicit capacity and several model families. This fork exposes token-based KV capacity, automatic KV headroom and WDDM budgeting. These are different contracts, not equivalent implementations. Actual startup capacity measurements determine the findings here. |
| Neroued `45f742877550d550d4359ecdb370a06cdaaf0aac` | Not transplanted. Important correction to the older report: `GenerationService::prepare` submits a request, but does **not** wait for physical Engine admission. This fork permits queued streaming requests and has SSE heartbeat/progress semantics. Merely finding `prepare` does not prove that upstream's pre-header-admission behavior exists. Queue timeout/error and cancellation must be assessed against this fork's advertised streaming contract. |
| Neroued `a062e0165ef2c26e433b92c639b5484539a48765` | Linux socket reuse behavior and successor-listener tests already exist. Their Linux-only assertions cannot be claimed executed by a Windows run; Windows transport behavior is tested separately. No duplicate fix. |
| Neroued `704fa68da9addc5e353ba14f10004d4c9829a1f0` | The ordinary completion, active cancellation and active failure paths here already remove the slot and publish runtime stats before completing the response. The source upstream has a different scheduler/resource ownership model. No broad terminal-lifecycle rewrite. This does not establish equivalent ordering in every exceptional path. |
| Don-Chad `931bd49ba332f48fad4b5230eec5a95ec5701290` | Already present: Chat nonempty `prompt_cache_key` grants an end-of-prompt private boundary and keys allow the shared 256-byte limit. PR #10 adds streamed/aggregate tool-history parsing coverage. No duplicated implementation. |
| Don-Chad `e5dac1437cc7aa9e941604424022c68c579dcc1e` | Not transplanted: its shared-prefix export scan/sink/backoff implementation does not exist here. The hybrid cache persists Host-tier blocks/snapshots with its own framing and model/geometry fingerprint. No dead compatibility implementation introduced. |

## Measurement-tool fixes

- Commit `aaa45c9c5261871dd0dbaff54f8d82b01cc0a681`: a failed model process now
  makes the graph sweep fail even when it printed a memory observation; GDN-FP16
  is passed and recorded; sweep comparison rejects mismatched GDN precision;
  the PowerShell driver accepts a verified Python interpreter; Windows uses
  `cuda,nvtx,wddm` instead of unsupported Linux `osrt`.
- The current CLI prints separate `CUDA Graph used` / `CUDA Graph allowance`
  fields. The runner formerly expected an obsolete combined line. It now consumes
  the current fields, preserves complete output per run and refuses to substitute
  the allowance when usage is explicitly unmeasured.
- The profiled benchmark includes a combined 2048-token prefill + 128-token decode
  and writes native JSON beside the trace for phase attribution. CUDA Graph node
  tracing is requested explicitly. Profiled throughput is not an unprofiled gain.
- `rtx4090_cache_audit.py` exercises the HTTP product path, records actual token
  reuse and prefill timing, distinguishes aggregate latency from SSE TTFT, retains
  output differences and checks that complete SSE responses include `[DONE]`.

## Interpretation rules

Graph allocation deltas and `nvidia-smi` memory are device-wide observations.
CLI process wall duration includes model load, Engine startup and a one-token
generation; it is **not** isolated graph-capture time. Kernel duration sums may
double-count overlap. CUDA API wall durations include waits and are **not** CPU
instruction time. CPU scheduling events alone do not establish complete CPU
profiling. Occupancy, register pressure and bandwidth require Nsight Compute or
equivalent hardware counters; they cannot be inferred from Systems durations.

The bundled benchmark corpus is repetitive and its own manifest warns of very
high speculative acceptance. Its trace can identify kernel costs for that workload,
but cannot establish general Qwen Code decode throughput. The live HTTP audit
uses explicit coding prompts and retains the complete requests locally.

## Execution results

### Initial build and incomplete first test run

The full CUDA 13.4 / SM89 build completed all 1144 steps. The binary contained
native `sm_89` cubins (verified using Toolkit 13.4 `cuobjdump --list-elf`). The
pre-existing October 22 executables were not used for inference measurements.

The first CTest campaign began with 371 registered tests, using the old cache's
Python 3.13 selection. It was stopped after test 206 began, to investigate and
repair known failures before a corrected full rerun. This is an **incomplete**
campaign, not a passing full suite. Known failures and interventions:

1. `ninfer_multi_gpu_test`: fabricated Linux sysfs directory names contain `:`
   and cannot be created on Windows. Extracted interpretation of represented
   sysfs contents into a pure production parser. Windows now tests that parser
   and refusal decisions; Linux retains actual sysfs fixture lookup/normalization.
   Added CRLF, absent BAR1 and malformed resource cases. A separate native MSVC
   compile/run of the affected source and test passes (`peer-test.log`).
2. `ninfer_chat_templates_test`: old Python 3.13 had no Jinja2. Python 3.11 has
   been selected explicitly in CMake and its test dependencies installed.
3. `ninfer_qwen4_exp_ngram_writer_interop_test`: old Python 3.13 had no pytest;
   the conversion fixture also needs CPU PyTorch and Safetensors. Installed
   torch 2.15.0+cpu and safetensors 0.8.0 for Python 3.11. No production checkpoint
   or user model was downloaded or converted; the test converts its miniature fixture.
4. `ninfer_qwen4_exp_hybrid_experts_test`: mixed native CPU/DMA experts completed
   the eager T=16 call but stalled in its following graph replay. Earlier CPU-only
   graph/cancellation and mixed T=1/T=3 cases completed. Noninvasive CDB stacks
   show the caller and supervisor blocked in CUDA stream synchronization while
   board utilization is 100%. The audit terminated **its own** test process after
   170.37 seconds. An isolated run with `CUDA_MODULE_LOADING=EAGER` reproduces the
   same location; it was also terminated and is not counted as passed.

The first campaign already passed layout, OpenAI schema, HTTP transport, request
logging and Qwen3.5 context-store tests. Those successes do not erase the failures
or the unexecuted remainder of the campaign.

### Qualified Windows hybrid exchange correction

The diagnosed cycle is a captured H2D payload copy queued after the graph's
prepared-wait kernel, ahead of the supervisor's H2D expert DMA whose completion
releases that wait. The correction keeps the small table, partition and
CPU-result copies on the compute queue using an exact byte-copy kernel. Only the
Windows hybrid expert path selects this transfer. Core owns the allocation and
physical copy; the model retains expert composition and partition policy. No
expert arithmetic or stored representation changes.

Qualification adds exact byte and guard comparisons for eager and graph transfers
of 1/255/256/257/4099/1048593 bytes, graph replay after source changes, and source
bounds rejection. The existing hybrid FP64, fixed-partition bitwise replay and
stalled-worker cancellation matrix remains enabled. The isolated matrix passes,
and the focused CTest matrix passes it in 2.82–2.84 seconds. The exact-copy test
also passes. This establishes completion/copy/numerical correctness for those
cases, not an offload throughput improvement. Windows native hybrid exchanges
now issue up to three extra copy kernels instead of the corresponding H2D nodes;
the throughput cost on a real offloaded Qwen4Exp model is unmeasured. The target
dense Qwen3.8 workload does not execute this native expert path.

### IOCP thread-handoff deadlock and Windows failure fixture

The second full-suite attempt reached another existing hang in the standalone
`ninfer_expert_delivery_test` (not part of the `ninfer_tests` process). Its own
process was terminated after 226.29 seconds and the campaign was later stopped
when it reached test 206. This attempt is also incomplete; `ctest-full.log`
preserves its exact subset. In retrospect this standalone test was still running
in the first parallel campaign too; its incomplete result must not disappear
behind the other tests that completed.

Initial stack snapshots happened to catch transfer APIs, but were insufficient
to identify the cause. Both eager module loading and direct Driver H2D dispatch
failed to cure the isolated case. A trial mapped-kernel disk upload also failed.
**All those disk-transfer experiments were removed.** Instrumentation then
identified a stall in `read_now` after the staging-completion event, before any
new upload, exposing the actual IOCP handoff problem.

The queue was created with concurrency 1. Its startup/priming caller remains a
runnable thread associated with the completion port while spinning on a
GPU/host handshake. IOCP therefore cannot dispatch the service worker's read
completion. This matches Microsoft's [documented runnable-thread scheduling
rule](https://learn.microsoft.com/en-us/windows/win32/fileio/i-o-completion-ports).

The regression first primes a queue on one thread, transfers its next batch to
a worker, and keeps the old caller runnable. It fails on the old implementation
with `reader handoff blocked behind the prior runnable caller`; joining the
worker then permits the old IOCP to drain, so the test fails without hanging.
With the fix it passes buffered and direct reads at depths 1, 5 and 64, retaining
exact byte comparisons and failed-read drain/retry checks. The IOCP no longer
imposes an artificial runnable-thread limit; caller-owned batch serialization and
bounded slots/bounce buffers remain. It does not create extra I/O worker threads.

After that repair the full ExpertDelivery matrix exposed a further Windows
fixture error: truncating a file after share-read handles are open is forbidden.
The test now truncates before opening those handles while retaining the original
extents, so the first actual service read still fails asynchronously and must
release the GPU wait. No production file-sharing policy was weakened.

The isolated disk/CPU-share test now passes with zero byte/echo mismatches.
The complete host/disk/cache/CPU-share/elastic/lent/prefetch/error-control matrix
passes in 2.95 seconds. The four affected CTests (file queue, ExpertDelivery,
exact mapped copy, native hybrid experts) pass in 6.13 seconds overall. These
completion times distinguish the fixed hangs; they are not end-to-end model
throughput comparisons. Fix commit: `746c44fc`; hybrid fix: `0a40ca01`; portable
peer/layout checks: `b04550ac`. A new complete 371-test campaign followed.

### Third complete campaign: dependency and architecture findings

The third campaign runs all 371 registrations serially with Python 3.11 and a
1200-second per-test deadline. Its n-gram writer interoperability test initially
failed because Python 3.11 lacked `gguf`. Installed gguf 0.19.0; the actual CTest
then passes in 1.92 seconds (`ngram-interop-recheck.log`). This does not retroactively
make the earlier campaign pass.

The concurrent TMA share test also failed: it directly requested FP8 A8 on an
SM89 build, bypassing native preparation's supported A16 policy. The production
exception correctly said that this path requires sm_100a/sm_120a. The harness now
selects A16Only on compatibility builds for both workspace sizing and execution,
retaining every shape, both streams, 20 concurrent rounds and exact output
comparisons. Blackwell builds retain the original A8/A4 TMA cases. This is a
portable concurrency qualification, **not** evidence about Blackwell TMA stages.
No failed numerical comparison was tolerated or test disabled.

The live Chat benchmark does not send `cache_prompt: false`: that extension is
implemented for text completions, not Chat. Disabling it per Chat request would
silently produce an invalid control. Separate servers with `--no-prefix-reuse`
provide the disabled control; cold/warm identical-prompt pairs provide the local
comparison. `prompt_cache_options.mode=explicit` also does not disable Engine
structural prefix reuse and is not used as a false negative control.

### Real artifact cache qualification

The third full campaign completed in 3451.33 s: **337 passed, 32 skipped,
2 failed**. Missing conversion dependency and the unsupported TMA test request
are the two failures, subsequently repaired and individually rerun. That campaign was not accepted as green; subsequent complete campaigns are recorded below. Skips are optional real artifacts or a
build without Blackwell TMA; they are not passes.

The first actual rk4v4 / FP16-GDN / DFlash2-K5 restore test failed its fixture's
hard-coded 1024-token assertion: both Device and Host reused **1280** tokens,
with exact generated-token equality and actual Host restoration. The device's
prefill alignment selects 640 from requested 512, so 1024 was a private schedule
assumption, not the cache contract. The test now requires identical nonzero reuse
through both tiers, bounded by the independently constructed 1400-token common
prefix, actual Host block/image restores and exact 24-token output equality.
No production prefill alignment was disabled and no output tolerance changed.

The persistence fixture was extended to DFlash2. Its first extension accidentally
left the smaller-tier subcase on the non-speculative backend, correctly causing
fingerprint rejection. This audit-introduced fixture bug was fixed by applying
the selected backend to every subcase. The rerun passes whole-tier restart,
exact output/frontier equivalence to an uninterrupted Engine, incompatible build
identity refusal, partial restore into a tier one slab smaller, stop while active,
new-work refusal after stop and both abandoned-save cases. `turns-protocol` also
passes. Logs preserve both failed attempts and the successful rechecks.

### Live Qwen Code-style Chat workload (completed)

`cache-run2` is the authoritative HTTP campaign. It repeats the whole campaign
after fixing an audit-tool bug: a growing Python message list had retroactively
changed the recorded payload of earlier long-session requests. The first run's
original responses/timings remain local, but its long-history payload records are
not reproducible evidence. The second run snapshots every payload when recorded.
Both runs' functional assertions pass; neither server required forced termination.

This is the real OpenAI Chat HTTP path with representative coding/tool histories,
**not** an invocation of the Qwen Code client executable. Both servers use the
specified artifact, DFlash2 K=5, rk4v4, FP16 GDN, two lanes, requested chunk 1024,
context/KV capacity 32768. Hybrid has a 256 MiB Host-tier budget; the control uses
`--no-prefix-reuse`. Each arm records 48 scenarios, plus queue and invalid-key
probes. `cache-summary.json` retains every measured pair and case without replies.

| Five identical-prompt pairs per transport | Hybrid cold min/median/max | Hybrid warm min/median/max | Disabled repeated min/median/max |
| --- | --- | --- | --- |
| SSE client TTFT, ms | 1045.955 / 1059.447 / 1062.962 | 23.192 / 37.769 / 38.893 | 993.466 / 994.562 / 1018.948 |
| Aggregate whole response, ms (not TTFT) | 1223.179 / 1239.345 / 1282.867 | 205.314 / 217.894 / 232.737 | 1150.044 / 1180.252 / 1187.248 |

For all ten Hybrid cold/warm pairs the actual Engine counters show cold
3536 computed / 0 reused and warm **7 computed / 3529 reused**. The five SSE
cold prefill intervals are 1042.704–1046.059 ms, warm 20.992–22.407 ms.
The paired cold/warm SSE TTFT ratios are **27.240, 27.905, 45.833, 27.879,
32.074** (median 27.905); worst warm latency is 38.893 ms. This measures reuse
of already computed input within this workload, not a kernel optimization or
an improvement over an optimized previous fork. All ten cold/warm outputs match
exactly inside each mode. Disabled repeats compute the complete prompt again.

**Observed cold-request cost:** Hybrid cold median SSE TTFT is 63.262 ms
(6.35%) higher than disabled cold, and aggregate cold median response is
58.800 ms (4.98%) higher. The mode groups run sequentially and allocate different
cache resources, so these are observed cross-mode differences, not an isolated
causal overhead estimate. They are not dismissed as noise. The warm-pair benefit
does not erase this cold-request result or the additional observed memory usage.

**Unfavorable output finding:** across modes, 7 of 44 identical recorded request
payloads produce different greedy text: `False/4/cold`, `False/4/warm`,
`False/second-tool-turn`, `long-session/0`, `after-cancel`,
`after-prefill-cancel`, `after-queue-stress`. Therefore full cross-mode output
identity is **not established**. Cache taps can alter prefill decomposition and
FP16 state roundings; that is a possible explanation, not a separately isolated
cause here. Four subsequent long-session payloads also differ because each arm
continues its own generated history; cross-mode timing/output comparisons of
those four are not matched-prompt A/B evidence. No arithmetic or acceptance
criterion was changed to remove these findings.

Tool-call continuity works without adding a new user turn. In both aggregate
and SSE transport, the first tool turn reuses 3529 / computes 58; its repeat
reuses 3580 / computes 7. A second tool turn reuses 3580 / computes 65; its
repeat reuses 3638 / computes 7. Omitting `prompt_cache_key` on the repeated
history still reuses 3580: Hybrid content-addressed sharing is not a per-key
namespace. Changing the leading system prompt computes all 36 tokens with
zero reuse; a different key with that identical new content then reuses 29.
Two concurrent ALPHA/BETA sessions match their own fresh references exactly;
global counter deltas are deliberately not attributed per concurrent request.
Numeric keys are rejected with HTTP 400.

The Hybrid long session begins at 20613 computed / 0 reused tokens and
5976.388 ms TTFT. Its next three turns and retry respectively compute
61/61/45/30 and reuse 20606/20660/20714/20752, at TTFT
81.225/80.846/71.817/70.222 ms. These are measurements of its actual evolving
history; disabled-arm later histories are not interchangeable baselines.

A client disconnect after first content settles in 74.606 ms (Hybrid) / 74.149 ms
(disabled). A disconnect before first content during fresh long prefill settles
in 337.832 / 292.987 ms. The Hybrid retry correctly reuses only the committed
1280-token prefix and computes the remaining 19334. Both overload probes return
429; queued SSE timeout errors arrive after 2.043 / 2.017 s. Subsequent requests
work, final running/waiting/materializing/prefilling/decode-ready counts are zero,
and both server processes stop normally with exit 0. This covers client
cancellation and pending-queue timeout, not arbitrary active-generation deadlines.

Device-wide post-request memory spans 17286–17288 MiB (Hybrid) versus
16931–16940 MiB (disabled). This is approximately 346–357 MiB additional board
usage in these separate WDDM process runs, **not** a certified per-process cache
allocation. Actual host RSS and sub-request memory peaks were not sampled in
this campaign; the 256 MiB Host tier is a configured budget, not measured RSS.

## CUDA Graph capacity, startup and VRAM

All measurements use the exact artifact, native SM89 Release binaries at
`1922c64a8052d79dd35466891e96021ceafb9445`, rk4v4, FP16 GDN and requested prefill
chunk 1024. `graph-samples.csv` contains **all 45 samples**, including both
initial default-copy campaigns, not selected favorable samples. Each row records
return code, device-wide graph delta, planned allowance, wall duration, reported
Engine startup and the 100 ms sampled board-memory peak. All 45 processes exit 0;
all startup capacities are honored. Loads repeat in fresh processes; disk/driver
cache state is not reset or measured. Payload materialization uses direct I/O,
so these must not be described as guaranteed warm-page-cache or cold-disk timings.

The product automatically enables 15 copy-proposal tokens alongside DFlash2 unless
explicitly overridden. This was discovered in the actual native benchmark JSON:
the initial captures used eight 16-token context-copy verification rounds, with
120/120 copy drafts accepted. They are **not neural-only K5 traces**. Their
results remain separate. The driver now exposes `-NgramDraftTokens` (default 0)
and records it; the sweep passes the explicit width, and the comparator refuses
mismatched copy widths. This changes measurement configuration, not product defaults.

Primary **neural K5, copy width 0**, five starts per capacity:

| Context/KV tokens | Graph used MiB, all five | Planned allowance MiB | Sampled board peak MiB min/median/max | Engine startup s min/median/max | Whole CLI wall s min/median/max | Free after startup |
| --- | --- | --- | --- | --- | --- | --- |
| 8192 | 34 / 34 / 34 / 34 / 34 | 288 | 16119 / 16119 / 16120 | 3.8 / 3.9 / 3.9 | 4.49 / 4.50 / 4.55 | 8.46 GiB |
| 32768 | 42 / 42 / 42 / 42 / 42 | 384 | 16547 / 16547 / 16549 | 3.9 / 3.9 / 3.9 | 4.52 / 4.54 / 4.56 | 8.04 GiB |
| 65536 | 50 / 50 / 50 / 50 / 50 | 480 | 17115 / 17117 / 17119 | 3.9 / 3.9 / 3.9 | 4.56 / 4.58 / 4.60 | 7.49 GiB |

The explicit-capacity summary prints `KV capacity headroom 0 B`; that is the
policy's reserved headroom, **not** zero remaining VRAM. KV payload is 140 MiB,
560 MiB and 1.09 GiB respectively; total runtime reservation is 927.3 MiB,
1.41 GiB and 2.05 GiB. No startup memory bottleneck occurs under these conditions.
This tests allocation/capture plus a 13-token prompt and one output, not a fully
populated 65536-token inference session. Sub-100-ms peaks remain unmeasured.
CUDA's free-memory summary and NVML/WDDM board usage have different accounting;
they must not be treated as complementary parts of one physical-memory total.

Default **neural K5 plus copy width 15**, two complete campaigns of five starts:

| Context | Graph used MiB (range across all ten) | Allowance MiB | Run 1 / run 2 peak board MiB ranges | Engine startup s range | CLI wall s range | Free after startup |
| --- | --- | --- | --- | --- | --- | --- |
| 8192 | 86–86 | 576 | 16188–16190 / 16200–16202 | 4.2–4.3 | 4.66–4.78 | 8.38 GiB |
| 32768 | 106–106 | 768 | 16628–16633 / 16640–16642 | 4.2–4.4 | 4.71–4.89 | 7.95 GiB |
| 65536 | 126–126 | 960 | 17211–17212 / 17217–17218 | 4.3–4.4 | 4.81–4.90 | 7.39 GiB |

There is zero reported graph-delta spread at the CLI's display precision, while
board peaks vary by up to 14 MiB between default-copy campaigns. Desktop/WDDM
activity remains a confounder; the delta is not a dedicated per-process allocator
measurement. Allowance is a planning budget, not observed physical usage. Neither
the startup log nor the CLI wall timer isolates graph capture itself.

### Measurement failures and their disposition

The first phase-3 script completed both measurement stages but its family parser
failed to identify IQ4_XS. MSVC/CUDA names the template argument `(ggml_type)23`,
not `IQ4_XS`; the vendored stored-format enum identifies IQ3_S=21 and IQ3_XXS=18.
The parser now recognizes those typed casts, never arbitrary numeric tile sizes,
with a regression that distinguishes unrelated dimensions/precision enums.

That first script also printed a misleading success footer: assigning a local
`$LASTEXITCODE` after profiling shadowed later native command exit codes. Removed
that assignment; the profiler's saved result is checked directly. The entire
script was executed again successfully (`phase3-run2`), retaining the first
failure/log rather than relabeling it a pass. Its valid graph samples are reused.
A third complete script run explicitly measures the newly separated copy-width-0
configuration (`phase3-neural-k5`), not an opportunistically changed baseline.
All 25 graph/comparison/kernel/phase analysis Python tests pass.

## CPU, CUDA API and GPU observations

Requested **Nsight Systems 2026.1.3.425** was actually used for all three campaigns.
It reports that installed CUDA driver 13.4 is unsupported and collects through
its CUDA 13.3 libraries. Both the requested tool and the separately installed
**2026.5.1.161** supplement warn that not all CUDA/NVTX events may have been
collected, system profiling may not have started correctly, and no System events
were collected. The newer tool removes the driver-version warning but **does not
certify complete tracing**. Scheduling rows exist; this does not establish
complete CPU instruction-execution coverage or WDDM residency/preemption coverage.

The following are measured intervals in the captured target stream/process,
**not CPU busy time, SM busy time, or proof of GPU idleness**. Exact workload:
combined pp2048+tg128, one discarded warmup, one measured repetition, neural K5,
copy width 0, rk4v4, FP16 GDN, context8192. The corpus is repetitive. The neural
capture accepts 103/120 drafts (85.83%) in 24 rounds plus one fallback; it records
zero n-gram rounds. Profiled native decode is 238.81 tok/s, not an unprofiled
throughput claim. The earlier default-copy capture is 100% copy acceptance and
562.70 tok/s; those different proposal/workload paths are not an optimization A/B.

| Interval, ms | 2026.1.3 prefill | 2026.1.3 decode | 2026.5.1 prefill | 2026.5.1 decode |
| --- | --- | --- | --- | --- |
| NVTX host phase wall | 550.741 | 536.544 | 549.481 | 536.706 |
| Captured kernel elapsed union | 545.304 | 512.322 | 543.966 | 513.273 |
| Captured kernel+transfer union | 545.451 | 512.462 | 544.110 | 513.416 |
| Outside captured activity | 5.290 | 24.082 | 5.370 | 23.290 |
| Host submit wall (overlaps device execution) | 224.815 | 24.795 | 214.824 | 24.848 |
| Host wait wall | 325.810 | 511.227 | 334.555 | 511.440 |
| Longest uncovered interval, including boundaries | 0.091 | 1.123 | 0.187 | 1.065 |

Missing events can account for uncovered intervals. Calling the 24.082 ms
"GPU waiting for the CPU" would be unsupported. Actual CPU execution time,
complete launch-gap causes and complete GPU utilization are **not measurable
from this evidence**. Engine-native and NVTX phase scopes differ by +1.151 ms
for prefill and -0.553 ms for decode in the requested-tool capture; both timers
are retained rather than forced to agree. The compatible supplement differs by
+0.688/-0.455 ms. These are different measurement scopes, not a numerical tolerance.

`cuda-api-phases.json` clips Runtime API intervals to the measured NVTX phases,
excluding `cudaProfilerStart` setup (which alone occupies 1.419 s in the initial
whole-capture API summary). For neural K5 under 2026.1.3, prefill has 2628
`cudaLaunchKernel` calls totaling 217.221 ms and 518 `cudaLaunchKernelExC` calls
at 2.488 ms. Decode has **25 `cudaGraphLaunch` calls, 23.724 ms summed API wall**,
26 stream synchronizations totaling 510.981 ms and 25 ordinary launches totaling
0.520 ms. The 2026.5.1 supplement measures graph API wall 23.888 ms and stream
synchronization 511.211 ms. These intervals include driver waits and scheduling;
none equals CPU instruction execution. Host submission and device work overlap,
so their wall durations must not be added to infer total time.

At 100-ms board telemetry resolution each neural phase has five samples. Requested
tool capture: prefill graphics clocks 2580–2685 MHz, decode 2580–2670 MHz;
memory clock 10501 MHz; reported phase temperatures 45–60 C, board memory 16164 MiB.
Power samples cover only 164.47–304.82 W, so they do not establish peak
instantaneous power. Clock-event masks include 0x4/0x400. The compatible capture
has similar coverage limits. No thermal or CPU bottleneck is inferred from them.

The five board GPU-utilization samples in the requested-tool prefill phase are
1/28/28/28/28%; decode is 28/100/100/100/100%. The supplement reports prefill
1/1/1/100/100% and decode100/100/100/97/97%. These are retained rather than
averaged into a misleading phase-utilization claim. NVIDIA defines this metric
over a trailing product-dependent 1/6–1-second sample period, not the 100-ms query
interval or instantaneous SM occupancy ([nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)).
Thus the board values cannot establish utilization separately for these roughly
half-second phases; the individual Nsight Compute counters answer a narrower question.

## IQ4_XS, IQ3_S and IQ3_XXS kernels

The corrected Windows parser attributes **captured summed kernel duration**, not
elapsed wall time. Requested-tool neural capture totals 1057.790 ms; its family
shares are IQ3_S **28.7827%**, IQ4_XS **18.0324%**, IQ3_XXS **16.3802%**.
The compatible supplement reports 28.4437%, 18.2780%, 16.4676% respectively.
These are two partial captures, not a claim of complete kernel coverage or a
representative outcome distribution. The largest captured families support
prioritizing IQ3_S as well as IQ4_XS. Historical 18.49% IQ4_XS is a different
capture/workload and is not a gain/regression baseline for these percentages.

**Nsight Compute 2026.3.1**, administrator privileges, exact same combined Engine
workload, neural K5, one first matching prefill MMQ and one first matching T=6
verification Vec kernel per format. LaunchStats/Occupancy/MemoryWorkloadAnalysis/
SpeedOfLight, kernel replay (nine passes), default cache flushing, clocks left
unmodified. Kernel replay warns that it backs up device memory in host RAM.
These counters describe six individual launches, not every model shape or
unprofiled inference. `kernel-counters.json` preserves exact kernel names, grids,
blocks, metric values and units; large `.ncu-repz` captures remain local.

| Format / sampled route | Kernel duration us | Registers/thread | Achieved occupancy % | DRAM GB/s | Register / shared-memory CTA limit |
| --- | --- | --- | --- | --- | --- |
| IQ4_XS prefill MMQ | 501.696 | 254 | 16.665 | 88.892 | 1 / 1 |
| IQ3_S prefill MMQ | 574.080 | 254 | 16.665 | 66.053 | 1 / 1 |
| IQ3_XXS prefill MMQ | 359.264 | 238 | 16.660 | 62.998 | 1 / 1 |
| IQ4_XS verification Vec T6 | 33.056 | 134 | 23.822 | 849.677 | 3 / 8 |
| IQ3_S verification Vec T6 | 30.464 | 128 | 31.001 | 759.042 | 4 / 10 |
| IQ3_XXS verification Vec T6 | 20.160 | 101 | 29.322 | 614.997 | 4 / 8 |

Different rows execute different matrix shapes; these durations are **not a
format-versus-format speed comparison**. Prefill MMQ has a simultaneous register
and shared-memory one-CTA limit; reducing registers alone may not raise occupancy.
The verification samples sustain much higher DRAM throughput and have a different
constraint. Low occupancy alone is not proof of a throughput problem.

Two NCU selection attempts collected **no kernels**: the first filtered MMQ in a
tg128-only workload whose actual route was Vec; the second sought an int16 MMQ
verification route after disabling copy proposals, while the actual K5 route is
Vec T6. Both failures/logs remain local. Filters were corrected from the actual
Systems kernel inventory, and the matched pp2048+tg128 six-capture matrix now
exists. The first successful export also exposed the newer `.ncu-repz` extension;
the helper initially looked for `.ncu-rep`, then reused that valid capture rather
than rerunning it. An exit code without an actual report is never counted as
counter evidence.

## Unprofiled alternating A/B: existing Graph versus Eager

Identical 80-function Python coding prompt, 128 requested output tokens, greedy,
no thinking, DFlash2 K5, product-default copy width 15, rk4v4, FP16 GDN,
context/KV8192, requested prefill 1024; fresh process per arm, five paired trials,
order reverses each pair. Disk/driver cache state is not controlled. No GPU tests/profile run overlaps
these trials. Desktop applications remain open. Both binaries are the same
1922c64a executable, with only `--no-cuda-graph` selecting the Eager arm.

| Pair / order | Eager tok/s | Graph tok/s | Graph/Eager | Engine startup Eager / Graph s |
| --- | --- | --- | --- | --- |
| 1 E/G | 163.2 | 190.4 | 1.16667 | 3.3 / 4.1 |
| 2 G/E | 162.7 | 190.1 | 1.16841 | 3.4 / 4.1 |
| 3 E/G | 163.3 | 190.3 | 1.16534 | 3.4 / 4.1 |
| 4 G/E | 162.9 | 184.7 | 1.13382 | 3.4 / 4.1 |
| 5 E/G | 163.8 | 190.5 | 1.16300 | 3.3 / 4.1 |

All ten subprocesses exit 0; all five pairs have **exactly identical generated
UTF-8 text**. This is text equality, not a token-ID oracle. Paired median Graph
advantage is **16.534%**, range **13.382–16.841%** (5/5 positive). Pair 4 is
materially slower than the other Graph trials (184.7 versus 190.1–190.5); its cause
was not isolated and it is not discarded as noise. Endpoint graphics clocks span
2670–2685 MHz during most trials, with initial idle210 MHz; recorded temperatures
40–57 C. Endpoint sampling does not establish continuous clock/power isolation.

**Cost:** Graph Engine startup is 0.7–0.8 s longer in every pair (coarse 0.1-s
startup log), including load/capture and other Engine work. That is an observed
startup tradeoff, not isolated capture latency. Graph already is the optimized
production implementation; this comparison establishes its benefit over Eager
fallback for this one workload, **not a new fork speedup**. No experimental SFU,
vector or T6 loop kernel option was enabled or promoted.

## Recommendations bounded by the evidence

1. Keep the existing Graph default for this measured coding workload: all five
   pairs are faster and text-equal, while accepting the measured startup cost.
   Long-lived servers amortize that cost. The collected fresh-process CLI wall
   times below already show a short-request latency cost; they are not cold-disk measurements.
2. Prioritize IQ3_S MMQ alongside IQ4_XS. Test a smaller tile / reduced live
   dequantization state only as a candidate that jointly lowers register and
   shared-memory constraints; validate packed inputs against independent FP64
   decoding and production-route oracles before alternating whole-inference A/B.
   No such candidate is implemented or declared faster here.
3. Treat Vec T6 memory traffic separately from MMQ occupancy. The measured
   verification kernels already reach615–850 GB/s; investigate reloads and reuse
   at real shapes before assuming an instruction-only optimization will help.
4. Always record `ngram_draft_tokens`. Copy width 0 reduces observed graph usage
   from 86/106/126 to34/42/50 MiB here, but also removes the exceptionally fast copy
   path on repetitive output. This is a capability/configuration tradeoff, not
   a free optimization; product defaults remain unchanged.
5. Investigate the greedy cross-mode cache output differences before promising
   universal output identity. Retain the exact Host/Device/persistence proofs and
   compare fixed prompt/chunk boundaries if isolating FP16 rounding effects.
6. CUDA Graph planning allowances substantially exceed observed deltas in this
   workload. A calibrated SM89 allowance could improve automatic capacity, but
   these single-model desktop observations do not qualify a new global constant.
7. Resolve Systems/Windows event-coverage warnings before claiming CPU execution
   cost or causal GPU stalls. Current API waits, kernel intervals and hardware
   counter samples answer narrower questions only.

## Reproduction and retained evidence

Use VS 2022's `vcvars64.bat`, CUDA 13.4, Ninja Release, architecture89 and explicit
Python3.11. The CMake cache/build logs and the production artifact inspection are
local. After building: `cmake --build build-rtx4090 -j`; final complete suite:
`ctest --test-dir build-rtx4090 -j2 --output-on-failure --output-junit PATH`.
Integration tests are run serially with the exact artifact and the environment
variables documented in `tests/README.md`; select the three scenarios listed above.
Do not set that artifact for unrelated MTP/other-family fixtures.

```powershell
$py = 'C:\Users\SPK\AppData\Local\Programs\Python\Python311\python.exe'
$model = 'C:\ninfer-all\models\Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer'
& $py profiles/bench/rtx4090-audit/run_live_cache.py `
  --server build-rtx4090/apps/ninfer-serve.exe --model $model --out NEW_CACHE_DIRECTORY
.\scripts\measure-rtx4090-regressions.ps1 -Model $model -Python $py `
  -Nsys 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe' `
  -Out NEW_MEASUREMENT_DIRECTORY -GraphRepeats 5 -NgramDraftTokens 0
# Add -NgramDraftTokens 15 for the separately identified product-default copy path.
& $py profiles/bench/rtx4090-audit/graph_ab.py `
  --cli build-rtx4090/apps/ninfer.exe --model $model --out NEW_AB_JSON
.\profiles\bench\rtx4090-audit\profile-kernels.ps1 -Out NEW_NCU_DIRECTORY
```

Small sample/counter/phase summaries and reproducible helpers accompany this
report. Full `.nsys-rep`, SQLite, `.ncu-repz`, request/reply JSON, tool installation
and failed/successful native logs remain under this local audit directory.
Existing models, user files, previous benchmark results and configurations were
not deleted. Only audit-owned test/server/telemetry processes were stopped.

## Later materialization corruption found by the complete rerun

The next complete 371-test campaign (`ctest-final.xml`, code7109c9f7) finished
in 2083.40 s with **338 passed, 32 skipped and one failure**:
`ninfer_artifact_materialization_test`. A progress update had incorrectly said
there were no failures because it inspected only recent progress; the final
whole-log/JUnit review corrected that statement. This campaign is not a pass.

The failure first appeared as `a selected piece in the panel layout is not its
permuted bytes`. Repeating the original standalone test reproduced it after
25 successful executions, then another repetition campaign failed in the plain
selected-upload check. This established that it was not merely panel formatting
or parallel CTest scheduling. The original panel comparison reused the production
permutation as its reference. It has now been replaced by an independent bytewise
CPU permutation, while preserving complete byte comparisons including padding.
A focused256-round selected-upload/panel stress failed on old code immediately;
it reported byte0 as0 instead of 49.

Root cause: `cudaMemset` cleared selected-object padding on the legacy default
stream, while payload H2D ran on `DeviceContext`'s nonblocking rank transfer stream.
There was no dependency. A late memset could erase weight bytes that the copy
engine had already uploaded. The materializer now collects bounded zero ranges
and queues their `cudaMemsetAsync` on the **same rank stream before the copies**.
The existing completion guard covers zeroing and uploads, is destroyed before
staging slots, and drains on error. A rank with no file ranges still drains its
zero tail. This is a generic materializer correction, reproduced on Windows;
it changes no quantization mathematics, representation or kernel route.

Fix: `b6c0e4d2416542ac32e2f474077cb9049852bb47`, pushed to the requested branch.
Four affected tests pass: materialization, independent panel permutation,
Python-writer/C++-reader interoperability and the new selected-upload stress.
The ordinary fixture additionally covers a transcode-only rank's encoded bytes
and zero tail. Ten stress repetitions pass (2560 rounds,24.20 s); these timings
are qualification duration, not an inference speedup. Earlier module-loading and
DMA experiments were not reintroduced.

There is one extra registered regression, so the post-fix full suite contains
**372 tests**. All earlier performance/cache/counter tables above identify their
pre-fix source (746c44fc for HTTP;1922c64a for graph/A/B/counters) and are retained.
Because startup materialization changed, the
three real-model scenarios, both HTTP cache arms,15 neural-K5 graph-capacity
samples, alternating Graph/Eager trials and six NCU launches were repeated
on b6c0e4d2 before the new full CTest. The later section records that evidence;
no old measurement is silently relabeled as belonging to the corrected build.

## Post-fix requalification — b6c0e4d2

All evidence in this section uses freshly rebuilt
`b6c0e4d2416542ac32e2f474077cb9049852bb47` executables after the materialization
ordering fix. It supersedes earlier measurements as the current implementation's
qualification; earlier records remain evidence about their stated pre-fix builds,
including all adverse findings. The complete 372-test rerun finished: **340 passed, 32 skipped, 0 failed** (2087.68 s).

The same three real-artifact cases pass again: `restore-exact-dflash2`,
`turns-protocol`, `persist-dflash2`. Both fresh HTTP server arms pass all 48
recorded cases plus queue and invalid-input probes and stop normally (exit0,
no forced termination). All ten short cold/warm output pairs match inside each
mode. Across modes the **same seven of 44 identical payloads still differ**;
the four later long histories still differ. Fixing the upload race does not
establish cross-mode output identity or isolate the cause of those variations.

| Current cache metric | Hybrid cold min/median/max | Hybrid warm min/median/max | Disabled repeat min/median/max |
| --- | --- | --- | --- |
| SSE client TTFT ms, n=5 | 1057.131 / 1061.198 / 1068.116 | 22.477 / 33.207 / 33.677 | 980.321 / 994.597 / 1023.641 |
| Aggregate response ms, n=5 | 1223.788 / 1242.232 / 1285.582 | 205.397 / 211.536 / 217.087 | 1150.216 / 1179.998 / 1181.958 |

Actual Engine work remains3536 computed/0 reused cold,7 computed/3529 reused
warm. SSE prefill is 1044.659–1050.426 ms cold and 20.969–21.605 ms warm.
Compared with disabled **cold** median, Hybrid cold SSE is 63.992 ms (6.42%)
slower and aggregate cold is 63.563 ms (5.39%) slower. These are sequential mode
groups, not isolated causal costs. Earlier cold costs and TTFT ranges remain in
the preceding table; no version-to-version speedup is inferred from this rerun.

Long Hybrid session TTFT is 5962.314 ms initially, then77.152/75.504/63.207/68.974 ms
on its three continuations/retry, with the same20606/20660/20714/20752 reuse and
61/61/45/30 computed-token deltas. Other-session content/key and tool histories
retain their earlier exact counter outcomes. Stream-disconnect settlement is
77.043 ms Hybrid /75.729 ms disabled; prefill disconnect is 292.918/309.585 ms.
Queued SSE timeout arrives after 2.042/2.016 s; overload is 429 and post-cancel/
post-timeout requests work. These remain represented cases, not bounds on every
request or active-generation deadline.

Post-request board memory is 17167–17172 MiB Hybrid and 16817–16826 MiB disabled,
a341–355 MiB cross-mode difference. The lower absolute board values than earlier
runs are not claimed as a materializer memory improvement: WDDM/desktop baseline
changed and per-process allocation/RSS was not isolated. Host budget remains256 MiB.

Current neural K5, copy width 0, five successful starts per capacity:

| Context | Graph used MiB, all 5 | Allowance MiB | Board peak MiB min/median/max | Engine startup s min/median/max | CLI wall s min/median/max | CUDA free after startup |
| --- | --- | --- | --- | --- | --- | --- |
| 8192 | 34 / 34 / 34 / 34 / 34 | 288 | 16002 / 16004 / 16006 | 3.9 / 3.9 / 3.9 | 4.52 / 4.53 / 4.56 | 8.46 GiB |
| 32768 | 42 / 42 / 42 / 42 / 42 | 384 | 16426 / 16428 / 16429 | 3.9 / 3.9 / 3.9 | 4.51 / 4.54 / 4.55 | 8.04 GiB |
| 65536 | 50 / 50 / 50 / 50 / 50 | 480 | 16994 / 16994 / 16994 | 3.9 / 3.9 / 4.0 | 4.55 / 4.57 / 4.62 | 7.49 GiB |

No startup/capture failure occurs. All60 graph-start samples across all campaigns
are retained locally, with the15 current samples in `graph-samples-b6c0e4d2.csv`.
No fully filled64K inference claim, per-process memory claim or pure graph-capture
latency claim follows from these startup probes.

Current unprofiled coding A/B (same workload/order/flags as above):

| Pair/order | Eager tok/s | Graph tok/s | Graph/Eager | Startup E/G s |
| --- | --- | --- | --- | --- |
| 1 E/G | 163.0 | 190.6 | 1.16933 | 3.4 / 4.1 |
| 2 G/E | 163.2 | 190.8 | 1.16912 | 3.3 / 4.1 |
| 3 E/G | 163.8 | 190.8 | 1.16484 | 3.4 / 4.1 |
| 4 G/E | 163.5 | 190.5 | 1.16514 | 3.3 / 4.1 |
| 5 E/G | 163.5 | 190.0 | 1.16208 | 3.3 / 4.1 |

All ten processes exit0; all five generated texts are exactly equal pairwise.
Paired median advantage **16.514%**, range **16.208–16.933%**,5/5 positive.
Startup still costs 0.7–0.8 s more with Graph. This remains optimized-existing-Graph
versus Eager fallback, not a before/after version comparison. The older184.7 tok/s
outlier and 13.382% worst pair are retained; their cause is unresolved.

Current required Nsight Systems 2026.1.3 emits the same driver13.4 and incomplete
CUDA/NVTX/System warnings. No additional newer-tool capture was needed: the
existing2026.5.1 supplement already establishes that changing tool version does
not certify complete Windows coverage. Complete CPU execution time and causal
GPU idle time therefore remain unmeasured for the corrected implementation too.

| Current captured interval ms | Prefill | Decode |
| --- | --- | --- |
| NVTX host wall | 555.591 | 531.388 |
| Captured kernel elapsed union | 549.964 | 508.917 |
| Captured kernel+transfer union | 550.111 | 509.057 |
| Outside captured CUDA activity; not proven idle | 5.480 | 22.331 |
| Host submit wall | 222.212 | 23.360 |
| Host wait wall | 333.269 | 507.543 |
| Longest uncovered interval including boundaries | 0.185 | 1.100 |

The new prefill wall is 4.850 ms higher and decode5.156 ms lower than the earlier
neural capture. These separate captures are not interleaved version A/B; neither
a gain nor a slowdown is causally established, and neither observation is
removed. Runtime API intervals clipped to NVTX show2628 ordinary prefill launches
at 164.653 ms summed wall and 518 ExC launches at 37.979 ms; decode has 25 Graph
launches at 22.373 ms and 26 stream synchronizations at 507.320 ms. These are API
elapsed intervals, not CPU instruction time. NVTX/native scopes differ by
+0.776 ms prefill and-0.520 ms decode. Requested100-ms board telemetry has 5 prefill
and 4 decode samples, with graphics 2535–2685 MHz, memory 10501 MHz, temperature
49–65 C, and power 175.31–318.59 W. Averaged board samples are not instantaneous
phase utilization. Profiled decode is 241.115 tok/s with 103/120 neural drafts
accepted,24 rounds, one fallback and zero copy rounds; not an unprofiled gain.

Captured summed kernel duration is 1059.026 ms; IQ3_S/IQ4_XS/IQ3_XXS shares are
**28.3967%/18.2503%/16.3734%**. Coverage remains partial. Six fresh Nsight Compute
launch captures use the same matched workload and nine-pass replay settings:

| Format/route | us | Registers/thread | Achieved occupancy% | DRAM GB/s | Register/shared CTA limit |
| --- | --- | --- | --- | --- | --- |
| IQ4_XS MMQ prefill | 501.600 | 254 | 16.665 | 83.745 | 1/1 |
| IQ3_S MMQ prefill | 577.472 | 254 | 16.665 | 66.775 | 1/1 |
| IQ3_XXS MMQ prefill | 361.120 | 238 | 16.660 | 70.240 | 1/1 |
| IQ4_XS Vec T6 verification | 32.768 | 134 | 23.821 | 853.535 | 3/8 |
| IQ3_S Vec T6 verification | 30.208 | 128 | 30.961 | 752.479 | 4/10 |
| IQ3_XXS Vec T6 verification | 19.680 | 101 | 29.486 | 633.678 | 4/8 |

Different actual matrix shapes still preclude format-versus-format timing claims.
Recommendations remain: jointly target MMQ register/shared-memory pressure,
consider Vec memory traffic separately, retain Graph for this measured workload
with its startup cost, isolate cache-mode numerical variations, calibrate graph
allowances more broadly before changing defaults, and resolve Windows tracing
coverage before attributing CPU/GPU stalls. No experimental kernel is promoted.

Current small evidence files carry the b6c0e4d2 suffix; earlier files retain their
original source scope. The final complete-suite status below is verified against both the finished JUnit and full log, including every skip and failure.

### Whole-process A/B latency cost (retained adverse result)

The A/B helper also measured CLI process wall time, not only decode throughput.
Those existing measurements must not be omitted. Current b6c0e4d2 pairs:

| Pair | Eager whole CLI s | Graph whole CLI s | Graph minus Eager s | Graph wall increase% |
| --- | --- | --- | --- | --- |
| 1 | 5.466837 | 5.706025 | 0.239188 | 4.375 |
| 2 | 5.416752 | 5.707750 | 0.290998 | 5.372 |
| 3 | 5.447623 | 5.725418 | 0.277795 | 5.099 |
| 4 | 5.472277 | 5.730744 | 0.258467 | 4.723 |
| 5 | 5.397651 | 5.719381 | 0.321730 | 5.961 |

Graph is slower end-to-end in 5/5 fresh-process short CLI trials: median added
wall0.277795 s (range 0.239188–0.321730), paired median wall ratio1.050994.
The earlier 1922c64a campaign also has 5/5 higher Graph process wall, with
0.202704–0.312006 s added (median 0.299718 s, paired ratio1.055004).
This is not a contradiction: measured decode is faster, while startup and other
process costs differ. The data do not isolate capture itself. Disk/driver cache
state is not controlled, so this is not a universal cold-start latency estimate.
For this represented short one-shot workload Eager wins total latency; a
long-lived server amortizes initialization and benefits from the measured decode
path. No default is changed and no unmeasured token-count crossover is promised.


## Final validation and delivery

The b6c0e4d2 orchestrator exited **0** after the three real-model cases, both
48-scenario HTTP arms, the required profiling script, five alternating Graph/Eager
pairs, six NCU captures, and the complete CTest run. Final native suite:
**372 registered = 340 passed + 32 skipped + 0 failed**, **2087.68 seconds**.
The complete log has no non-skipped failure; finished JUnit independently confirms
these counts. This supersedes, but does not erase, both failed complete campaigns
and the earlier incomplete campaigns. No failing test was disabled.

Layout/allocation, artifact upload, OpenAI Chat/schema, request lifecycle,
HTTP streaming, context-store and available CUDA regression tests are included.
The new selected-upload stress test passed within this full run as well as its
2560-round focused repeat qualification. The 25 affected local tool tests and
20 tests in the separate tooling CI are additional evidence, not extra native
passes. Main CI skips Draft PRs, so it is not a successful native CI validation.

### Exact native skips

The following 31 tests need optional explicit artifacts/fixture environments;
they were not counted as passes. The selected artifact's hybrid restore,
conversation and DFlash2 persistence cases were run separately and passed;
that does not qualify every optional fixture or remove these CTest skips.

- `ninfer_engine_logprobs_real_test`
- `ninfer_ngram_graph_planning_real`
- `ninfer_ngram_lifecycle_real`
- `ninfer_ngram_archive_real`
- `ninfer_ngram_thinking_real`
- `ninfer_ngram_stop_chat_real`
- `ninfer_ngram_concurrent_real`
- `ninfer_qwen_native_streaming_real_test`
- `ninfer_qwen3_5_loading_real_test`
- `ninfer_qwen3_5_dflash_prefill_real_test`
- `ninfer_qwen3_5_prefix_real_test`
- `ninfer_qwen3_5_hybrid_prefix_real_test`
- `ninfer_qwen3_5_score_real_test`
- `ninfer_qwen3_5_vision_cpu_real_test`
- `ninfer_qwen3_5_vision_overlay_real_test`
- `ninfer_qwen3_5_vision_workspace_test`
- `ninfer_qwen3_5_dflash2_real_test`
- `ninfer_qwen3_5_moe_real_test`
- `ninfer_qwen3_5_stages_real_test`
- `ninfer_qwen3_5_suspend_real_test`
- `ninfer_qwen3_5_dflash_real_test`
- `ninfer_qwen4_exp_slice_real`
- `ninfer_qwen4_exp_generate_real`
- `ninfer_qwen4_exp_engine_real`
- `ninfer_qwen4_exp_long_context_real`
- `ninfer_gemma4_assistant_real_test`
- `ninfer_gemma4_residency_real_test`
- `ninfer_gemma4_vision_real_test`
- `ninfer_flint_branches_real_test`
- `ninfer_deepseek_v41_vision_real_test`
- `ninfer_nvfp4_int8_real_test`

The remaining skip is `ninfer_nvfp4_tma_staging_race_test`: this SM89 build has
no Blackwell NVFP4 TMA route. Separately, MSVC does not support GNU linker
`--wrap`; its fault-injection subcases are compiled out and therefore **not**
represented by these 32 CTest skips or by executed checks. Supported VMM
materialization paths did execute. Linux-only socket assertions and physical
multi-GPU behavior are not established by this Windows single-card run.

### Repository and resource disposition

All implementation commits are on the requested feature branch and pushed to
`origin`, the uniquely identified spk-c/ninfer-all-rtx4090 fork. Reports and small
reproducible summaries accompany the implementation; large Nsight reports,
SQLite exports, request logs and benchmark raw files remain local. The supplied
model was not modified, replaced or regenerated. Existing applications were not
terminated. Own hung tests during diagnosis and own benchmark servers were
stopped; both final server arms exited cleanly. No experimental kernel option
was promoted and no numerical-equivalence claim is made beyond the stated tests.

Protected master remains `a85c597f0b96686197712c4118deb2647ba7a75c`; the release
tag object remains `4ddc76ae318b88ef6cfe99dcb9f3750ecd8b7f42`. The pre-existing
user script retains SHA256
`769CA68FD37868339AFF8EB87B7486FA4D9FCF8BEBD81BC12228AA9E736F55EE`.
Draft PR #10 remains open and unmerged. No release was created. The final report
commit changes documentation/evidence only; b6c0e4d2 identifies the executable
implementation validated above.

Machine-readable final counts, skip inventory, toolchain and boundaries are in
[validation-summary.json](validation-summary.json). Earlier adverse findings and
all unresolved numerical, tracing, memory-attribution and throughput limits
remain part of the acceptance evidence; this is not an all-model or all-platform
certification and does not assert a newly optimized production kernel.
