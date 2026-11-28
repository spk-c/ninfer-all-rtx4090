# RTX 4090 development status

## 2026-11-27

- Fork master was `d3299b73`. iamwavecut master was `4a714474`.
- [Sync PR #57](https://github.com/spk-c/ninfer-all-rtx4090/pull/57) includes upstream
  #501–#510 through merge commit `a5a90362`. Both merges were conflict-free and preserve
  upstream ancestry. The RTX 4090 device-profile entries are retained. New ternary
  64-column profile entries apply to RTX 5090. Shared Q4 gate/up changes at 33–64
  rows can affect Ada. Measure that route on Windows before claiming a gain.
- [Hub PR #60](https://github.com/spk-c/ninfer-all-rtx4090/pull/60) compares native
  filesystem path values for the cached artifact and requires a regular file.
  The full Hub test passed on Windows with MSVC 19.51 by linking the CMake-built
  test object into a standalone entry point. The aggregate CTest suite requires
  building the shared test dispatcher. In the v0.1.7 release workflow the package
  job failed and publishing was skipped. v0.1.8 had no tag at review time.
- [Monitor PR #61](https://github.com/spk-c/ninfer-all-rtx4090/pull/61) at `801c4c2a`
  adds bilingual prefill/decode P50/P95/P99 from successful requests. The Windows
  host smoke and installed-Chrome Playwright fixture passed. Hosted checks and
  live integration remain pending. Further v3 scope is tracked in #59.

## Resuming validation

- Windows worktrees are `C:/coding/ninfer-codex-sync`, `C:/coding/ninfer-hub-win-fix`
  and `C:/coding/ninfer-codex-monitor`. The original worktree has two pre-existing
  untracked HF-card items. The native SM89 Release build for the sync uses
  `C:/coding/ninfer-codex-deps/build-1`, CUDA 13.4.92 and MSVC 19.51. Check its
  status and CI before claiming build success. The local Ubuntu WSL registration
  points to a missing virtual disk, so use hosted Linux CI for Linux validation.
- The RTX 4090 uses Windows WDDM with driver 621.40. The explicit primary artifact is
  `C:/ninfer-all/models/Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer`.
  Recheck GPU occupancy and existing server processes before running inference.
- #56 has historical K5/C2 and K7/C1 graph-reservation samples. No new default follows
  from those samples. Start the synced binary at the existing defaults, test
  explicit graph allowance overrides, and compare output, memory, and throughput.
- Upstream #500 remains open. The suspect range includes paired KV stores,
  BF16/q8_1 projections and serialized expert-copy changes. Upstream #506
  measured a partial recovery for Inkling with its forced kernel-copy path
  on Linux RTX 4090. Reproduce on Windows before bisecting. Confirm that the same
  artifact and placement can load at both historical endpoints.
- Upstream #449 and experimental #487 remain open. Evaluate elastic admission and
  Inkling shared pages separately, including rollback, eviction and live leases.
- Run synced serving schema and sampler tests, including the late-Reasoning cases.
  Perform actual GPU smoke with the primary model before a release qualification
  claim. Longer context, DFlash2/DSpark/MTP/N-Gram, graph allowance, WDDM and wide
  decode work await measured baselines. No new inference speed or VRAM result yet.

## Screening and benchmark validation

- [Load benchmark PR #62](https://github.com/spk-c/ninfer-all-rtx4090/pull/62)
  is stacked on #57. It rejects premature SSE EOF or missing final usage and
  finish reason, writes errors to the report, and adds P95 and total request
  latency. Four Windows Python 3.11 tests pass. The upstream-only branch is
  `codex/upstream-serve-load-validation` at `1a50227b`, based on `4a714474`.
  Retarget #62 to master after #57 is merged.
- Baseline screening used the archived v0.1.4 SM89 binary at `a6b1138b528b`,
  the primary Huihui artifact, 8192-token context limit, one lane, rk4v4,
  DFlash2 K7 and N-Gram off. After one warmup, all three 128-token completions
  succeeded. Engine decode was 111.52–140.14 tok/s, prefill was
  1089.90–1178.14 tok/s on 41–42 computed prompt tokens, and client TTFT was
  35.68–39.41 ms. Peak device memory was 22621–22679 MiB.
  Independently generated nonces produced different prompts. These are
  screening observations and do not establish a speed or memory delta.
- Raw logs, hashes, launch arguments and CSV/JSON/Markdown summaries are at
  `C:/coding/ninfer-codex-deps/baseline-8192`. Use matched prompts in both arms
  of the candidate comparison. Exact output hashes and peak host RAM are pending.

## Follow-up verification

- Monitor #61 head `e7edce16` also includes the rate summaries and sample counts
  in opt-in numeric history and JSON/CSV export. Re-ran the Windows host smoke
  and native browser fixture successfully.
- #62's strict SSE client passed real RTX 4090 streaming against the archived
  v0.1.4 package. The upstream-only patch is pushed to its own branch.
- Verified archive SHA256, 7-Zip integrity and all six portable launcher help
  commands for the archived v0.1.4 package with CUDA Toolkit removed from PATH.
  The intended Windows release-tool test subset reported 25 passed, 6 skips
  and 8 passing subtests. The POSIX publication suite failed when invoked
  directly under Windows (including temporary-directory cleanup errors). It
  belongs to the hosted Linux release-tools job, which passed on #62.
- Experimental upstream #487 head `07da430a` is on `codex/elastic-kv` at
  `dc9c17df` in `C:/coding/ninfer-codex-elastic`. The standalone MSVC queue
  policy test passed. Run `ninfer_inkling2_paging_test` and
  `ninfer_paged_8lane_real_test` before integration. Set
  `NINFER_PAGED8_ARTIFACT` to an explicit compatible Inkling artifact.
- No Inkling artifact was present in the inspected model directories or
  standard Hub-cache location. The Huihui and Bonsai artifacts are present.
  Inkling BF16 DSpark, the #500 paired Inkling measurements, and real paging
  tests need the appropriate artifact. A prefill bisect has no confirmed
  local reproduction. On Windows, leave forced copy-engine settings out of
  the default graph path because serialized copy queues can deadlock.

## Phase 2 release validation

- `eb997a54` is the local octopus-merge candidate containing exactly the four
  reviewed PR heads. The pushed `ci/rtx4090-shard-test/v018-validation` branch
  triggered build-only package workflow 46062515081. This path creates no tag
  or release and stamps a test version. Master remains at `d3299b73`.
- Upstream advanced to `ce57d546` with #511 Gemma 4 tool formatting and quoted
  key parsing. It is outside the four-PR candidate and awaits separate validation.
- The initial narrow build omitted the ALL-only test DLL staging target. CTest
  then failed to start the dispatcher with 0xc0000135. Building the full graph
  staged the DLL closure and native executables. A targeted rerun on the same
  candidate passed all seven tests (Hub, three protocol suites, raw completion,
  sampling and speculative-round). The whole 418-entry CTest run was then
  started with Python 3.11 and a 360-second per-test timeout. Review
  `C:/coding/ninfer-codex-deps/release-validation-ctest-full.log`.
- The SM89 sync server `a5a90362` and archived v0.1.4 both ran the fixed-prompt
  pilot. Output text SHA256 matched at 512, 2048 and 8192 actual computed
  tokens with cache reuse disabled and 128 generated tokens. This one-pair
  pilot overlapped the build and establishes wiring and output parity only.
  Logs and CSV/JSON are in `C:/coding/ninfer-codex-deps/fixed-ab-pilot`.
- For pilot C1 K7, the effective default was 384 MiB and the sync server
  measured 127.29 MiB of graph storage with capacity 397312 KV tokens.
  The 768 and 512 MiB cases therefore increase the allowance for this point.
  The historical long-context points use different settings. No default change.
- The GPU campaign was queued locally behind CTest. It alternates A and B
  over five repeats and seven actual prompt lengths, then runs the C1 graph
  screening for K5/K7/K9/K11/K15 at default, 768, 512 and 384 MiB.
  See `run-fixed-ab.py`, `fixed-ab-5.log` and `graph-c1-screening.log` in
  `C:/coding/ninfer-codex-deps`. If the session ended early, resume the
  incomplete arm using a fresh output directory and preserve the earlier rows.
- Draft release notes are `C:/coding/ninfer-codex-deps/rtx4090-v0.1.8-release-notes.md`.
  No PR was merged as CI and review were still pending. Complete the new
  portable-package and real-GPU gates before any tag or publication.

### Release gate follow-up

- The full GPU-backed CTest run exposed two Gemma 4 BF16 comparison failures
  for H=5376, T=5, despite correct FP32 residuals. PR #63 checks the
  downstream FP64 normalization against the observable stored FP32 residual
  and independently checks that residual against the FP64 input oracle.
  No numerical thresholds or production routes were changed.
  Rebuilt test object plus a separately linked dispatcher passed on RTX 4090.
- The hosted build-only branch now points to `1b5313ad`, including the test
  correction. Workflow 46063235612 replaces the queued older build.
- A local SM89 Ninja portable build is running in
  `C:/coding/ninfer-codex-deps/build-portable-v018`. See `portable-build.log`
  and `portable-package-vcenv.log` in its parent. The native development
  server and its staged DLLs resolved all 135 recorded imports.
  This does not verify the separate portable package. The measurement
  harness also waits for Ninja to exit before collecting timed samples.

## Phase 3 restart

- Source remains pinned to `1b5313ad` for the v0.1.8 integration build.
  Upstream advanced to `75ad218d` (#511 Gemma tool calls and #512 Tess artifact
  docs). Neither change enters this campaign.
- At restart, the original 418-entry CTest was still running. It reported the
  known Gemma #178 failure and a 360-second timeout in #248
  `ninfer_softmax_attention_test`. The timeout does not establish a
  numerical discrepancy. The quantized depth test #251 subsequently passed
  in 305.82 seconds. The corrected full run is queued with 1200 seconds per
  test and the #63 patch. If a test still times out, isolate its subcase.
- Coordinator PID 11104 runs
  `C:/coding/ninfer-codex-deps/run-phase3-resume.ps1`. It waits for the
  existing CTest and build processes, builds the corrected dispatcher,
  runs CTest, audits the actual portable package if available, pins the
  integration server, then runs the five alternating A/B repeats, the C1
  graph matrix and retrieval. Each phase logs under the same directory
  using the `phase3-` prefix. The original Phase 2 benchmark waiter was
  replaced before it launched GPU work.
- The updated A/B harness records paired hashes, request and TTFT
  percentiles, sampled GPU utilization and device VRAM, peak process working
  set, draft acceptance and coverage. It keeps completed rows even when
  a later request fails and pairs by repetition rather than list position.
  Python 3.11 compilation and the partial-result/percentile fixture passed.
- Graph screening now uses explicit 384, 320 and 256 MiB for each C1
  K5/K7/K9/K11/K15. It tries 192 and experimental 160 only after a
  successful preceding run whose measured graph allocation leaves at least
  64 MiB under the next budget. This is a gate for experiments and carries
  no implication for defaults. The planner already prices distinct topology
  classes separately per batch and proposal width. Its measured per-class
  price applies on SM86 and SM120. SM89 needs its own data including peak
  configurations, Vision and adaptive and n-gram widths.
- Huihui reports native context 262144. Retrieval points go through a
  262016-token prompt with 128 output tokens reserved. Above the native
  window is a separate extrapolation test with RoPE strategy stated.
  Startup KV capacity is recorded separately from prompt length and quality.
- Prepared three independent branches on current upstream and pushed them
  to the fork after checking upstream source and PR history:
  `codex/upstream-windows-hub` (`a2d3871e`),
  `codex/upstream-stream-validation-nov27` (`d03838b7`) and
  `codex/upstream-gemma-residual` (`8289e9de`).
- PR #62 remains stacked on #57. PRs #57/#60/#61/#63 were open with
  validation pending and no recorded review decision. No release tag or
  master change. The local portable build and hosted build-only job remain
  pending. Keep the release notes labeled draft until package and GPU gates
  actually pass.

## Phase 4 validation handoff

- Kept the existing coordinator (PID 11104) and integration source `1b5313ad`.
  No new package build or overlapping GPU run was started.
- Portable candidate `ninfer-windows-x64-cuda13.4-sm89-1b5313ad3501.7z.001`
  passed independent outer SHA256, 7-Zip integrity, staged and extracted
  verification of 66 inner checksums, PE dependency audit (219 resolved
  adjacent/system/API-set imports) and all six launcher help/version calls
  on a Windows-system-only PATH. Archive SHA256 is
  `979913a7c235b366d40cee3a679c791ae27021786804ab57997221d86cd64014`.
  Logs: `phase4-portable-offline.log`, `phase4-extracted-verification.log`
  and `dist/v018-portable-dll-audit.csv` in the validation worktree.
- CTest #248 is `ninfer_softmax_attention_test` with no selector.
  Its dispatcher executes causal-cache (including ordinary wide decode),
  plain/packed, and context references. Relative to fork master, the
  relevant upstream change extends the accepted batch-row domain and
  adds fragmented-cache cases at up to 64 rows (BF16/INT8/RK8V4/RK4V4/FP8).
  The 360.24-second original timeout produced no case-local failure.
  This is an unlocalized timeout. The archived v0.1.4 package has no CTest
  dispatcher for a like-for-like timing comparison.
- The existing coordinator calls the updated `verify-v018-package.ps1`
  after the corrected full suite. That hook runs #248 by itself with
  `--timeout 1200 -V` and samples GPU VRAM/utilization every three seconds.
  It then checks the extracted release server on a system-only PATH:
  Chat Completions, SSE termination/usage, forced strict tool calling,
  constrained JSON, a real image, native monitor/request metrics and
  authenticated graceful shutdown followed by restart. Logs are
  `phase4-attention-isolated.log`, `phase4-attention-telemetry.csv`, and
  `phase4-package-gpu/checks.json`. The GPU integration script compiled
  under Python 3.11. Its live results are pending.
- Upstream is `b7df0a63` (#516, Tess Q8 tower documentation). The three
  contribution paths had no relevant change since `75ad218d`.
  Keep upstream PR publication and fork merges gated by review.

- The original Windows suite completed all 418 entries in 4169.85 seconds:
  375 passes, 41 CTest skips, one Gemma failure and one attention timeout.
  The full log plus per-test CSV and timing summary are preserved in
  `C:/coding/ninfer-codex-deps/phase4-ctest-original-archive`.
  Two independent related selectors passed: wide attention in 306.81 s
  and quantized-depth attention in 305.82 s. The corrected archive is
  produced after its own suite completes.

## Phase 5 validation checkpoint

- The existing coordinator remains PID 11104. The corrected dispatcher
  built successfully and the corrected full Windows run passed Gemma
  test #178 in 5.86 seconds. It is currently inside attention #248.
  During the wait, the test process consumed roughly 38 CPU seconds over
  39 wall seconds, with a point GPU utilization sample of 2%. This shows
  ongoing host activity. No kernel deadlock diagnosis follows from that sample.
- Audited all 41 original CTest skips. Three match the test host: Unix
  fork-only wait checks, the two-GPU binding check, and NVFP4 TMA on SM89.
  The other 38 require artifacts, fixture data or opt-in setup. The per-test
  log or source prerequisite is in
  `phase4-ctest-original-archive/skip-audit.csv`. The complete original
  CTest dispatcher log is also preserved as `LastTest.log` there.
- The existing queue still isolates attention with the 1200-second limit
  after the corrected suite, then runs the packaged GPU smoke before
  performance measurements. Do not start another test or benchmark.
- The A/B summary generator now reports absolute paired medians, percentage
  changes, the range of paired changes, comparable-pair counts, and hash
  matches per prompt size. Raw JSON/CSV, failure records and percentile
  summaries remain. Python 3.11 syntax check passed. No measured rows yet.
- Upstream still reports `b7df0a63`. PR #62 remained green and stacked.
  The other four tracked PRs had checks pending and no failed check or
  recorded review decision. Source, master and release tags are unchanged.

## ChatGPT continuation checkpoint (2026-11-27)

- Took over via SentinelX without interrupting coordinator PID 11104 or
  changing integration source 1b5313ad. Corrected CTest #178 Gemma passed
  in 5.86 s and #248 `ninfer_softmax_attention_test` passed in 433.98 s
  (original local timeout was 360.24 s). The isolated 1200 s run with
  telemetry remains queued, and full corrected CTest is not finished.
- Confirmed the long test currently has no dedicated CTest TIMEOUT
  property; the original local 360 s limit was an invocation setting.
  Do not change production kernels or general CI timeouts on this basis.
- Hardened local `C:/coding/ninfer-codex-deps/run-fixed-ab.py`: paired
  performance summaries exclude repetitions whose prompt token hashes
  differ; mismatches are persisted in failures.json and make the run
  return nonzero. Output hashes remain a separate equality check among
  matching inputs. All raw measurements are preserved.
- CPU-only regression `test-ab-prompt-pairing.py` in the same directory
  passed: one valid pair, one mismatched input with extreme synthetic
  throughput, and one missing B arm. The report retained all five raw
  rows and reported only the valid +10% pair. Used the SentinelX Python
  interpreter because its service identity cannot find the user's
  py -3.11 installation. No GPU or model was used for this fixture.
- Reviewed SM89 graph admission in startup.cpp: per-topology classes are
  already counted per batch and speculative width; fallback DFlash
  pricing uses 64/96 MiB by visible context, and measured class pricing
  is 24 MiB per 64 text layers on eligible devices. Keep SM89 defaults
  unchanged pending the measured graph sweep. A C1/K7 startup reading
  is not sufficient to generalize to other topology families.
- PR #62 remains green and stacked on #57; #57/#60/#61/#63 have
  outstanding hosted checks, with no failed job at this checkpoint.
  All remain open. No merges or release publication. The existing
  coordinator still owns the corrected full CTest, isolated attention,
  packaged GPU smoke, A/B, graph screening and retrieval in that order.

## Phase 6: completed native validation and resumed performance screening

- Corrected native Windows SM89 CTest (integration source 1b5313ad)
  completed at 23:33:46 on 2026-11-27: 377 passed, 41 prerequisite
  skips, no failures or timeouts; 3985.68 seconds elapsed. See
  phase4-ctest-corrected-archive/{summary.json,cases.csv,ctest.log}.
- The isolated attention #248 passed all causal, packed and context
  checks in 434.63 s (full-suite result 433.98 s); both runs used 1200 s
  timeout. This reproduces a runtime greater than the original 360 s
  invocation limit without any assertion failure. Logs and GPU samples:
  phase4-attention-isolated.log and phase4-attention-telemetry.csv.
- The freshly extracted, offline-verified portable SM89 archive passed
  all 11 GPU integration checks: initial start, Chat Completions, SSE,
  forced tool calling, strict structured output, Vision, Native Monitor,
  graceful stop, restart, chat after restart, graceful stop. See
  phase4-package-gpu/checks.json and phase3-portable-validation.log.
- The first A/B/graph/retrieval launch failed before any inference
  measurements with UnicodeDecodeError in Python 3.11 decoding localized
  Windows tasklist output as cp1252. Changed only the local measurement
  harness to check the ASCII ninja.exe name in raw bytes. Extended
  test-ab-prompt-pairing.py with invalid cp1252 byte 0x81 and reran
  successfully. Original failure logs remain available.
- Restarted only the remaining GPU benchmark sequence with
  run-phase6-bench-resume.ps1 (PID 35476, started 00:00:26 on 2026-11-28).
  The script locks its own run, verifies the pinned 1b5313ad version,
  uses the actual Python 3.11 executable, and writes new results under
  phase6-resumed/ rather than replacing original logs. No rebuild or
  repeated release validation. A/B must produce usable rows before
  continuing to graph and retrieval screening. C1 graph screening is
  at 512 input tokens; do not generalize those results to long context
  or adaptive/vision/concurrency configurations.
- A/B began with actual GPU inference on v0.1.4. At this checkpoint,
  five reference prompts completed (512 through 65536 computed tokens);
  no comparable new-build pair yet. Manifest: phase6-resumed/fixed-ab/
  manifest.json. NVIDIA driver 621.40. Baseline binary SHA256
  1fcb3f20934997a81d980843e5ee3a0767de1ff82ee58153c543ea843004ff97;
  candidate binary SHA256
  1a00976cf25b52c1e704d46dfd4406b3715d9451d686d17c5f55769b46ba0f98.
- PR #60 is fully green (19 successful CI jobs, six expected skips).
  #62 remains green and stacked on #57; #57/#61/#63 still have
  outstanding checks. No failed hosted checks at this checkpoint.
  All PRs remain open. No master, integration source or release tag
  changes. No release approval or performance claim yet.

## Phase 7/8: performance results, correctness diagnostics, profile candidates

- Phase 6 finished at 00:41:52 on 2026-11-28. All 70 fixed-input
  A/B requests completed; all 35 input pairs matched, and 30/35
  output hashes matched (5/5 through 131072, 0/5 at 176128).
  Median prefill deltas at 512/2048/8192/32768/65536/131072/176128:
  -7.76/+0.61/+2.61/+3.06/+8.92/+13.51/+16.01 percent respectively.
  The short prompt had +15.03% TTFT; investigate warmed short calls.
  Long-input gains have paired-run variability and are not generalized.
  The 176128 output difference remains a correctness investigation,
  not proof of quality regression. See phase6-resumed/fixed-ab/.
- All 25 C1 short-prompt graph configurations started and completed
  a request, but the measured graph delta was sometimes zero and
  varied considerably (K5 at 160 MiB reported 111.64 MiB).
  384 to 160 MiB increased physical KV capacity by 13120 tokens for
  each tested width. Do not equate this with validated context or
  treat it as a safe default. K5 at 160 MiB did not retain the 64 MiB
  margin in its actual observation. Updated the stopped phase6
  coordinator's future low-budget gate to reject a nonpositive
  previous reading; historical results and logs remain intact.
- Both 65536 retrieval requests processed successfully but failed the
  original strict startswith test (rk4v4/rk8v4, GDN FP16). That test
  used raw completion, forced 128 output tokens and stored hashes,
  not answer text. Larger cases were stopped by design.
- Started SentinelX job job_611f6e3186db at 00:52:52, executing
  phase7-diagnostics.py with the pinned A/B binaries. The first
  176128-token raw-output comparisons showed different wording
  between old DFlash2 and old nonspeculative decoding; both recalled
  all three objects in the synthetic prompt. The new DFlash2 output
  also retained the three objects. These examples do not establish
  general equivalence; the new nonspeculative arm was still running.
  The diagnostic matrix captures actual answers, compares legacy
  forced and EOS-aware raw retrieval with chat-template retrieval,
  isolates GDN/KV/speculation configurations, runs warmed 512-token
  pairs, and tests K5/K7 C1/C2/C4 at 384/256 MiB with concurrent calls.
  Long-chat tests record actual computed lengths and multiple positions.
- Prepared four candidate PowerShell launchers plus common launcher
  and profiles.json under C:/ninfer-all/profiles/v018-candidates.
  All four -Preview invocations passed under PowerShell 7 (the
  SentinelX service's legacy Windows PowerShell execution policy
  blocks scripts; process-local invocation only, no policy changed).
  Existing production scripts untouched. Candidate launch requires
  explicit experimental override; validated configurations are pinned
  by binary SHA256 and checked for active inference and free VRAM.
- Started job job_a3d349ac9aa1 for phase8-profile-tuning.py at 00:56:01.
  It is waiting for the phase7 completion marker, not running another
  GPU process. After the marker, it screens K5/K7 against ordinary
  decode, compares 1024/2048/4096 prefill chunks, checks actual
  262016-token rendered-template inputs with 128 output tokens
  reserved at three positions, and qualifies balanced coding with
  strict tool/JSON schema and real vision. Only profiles passing
  their specific parity/retrieval/feature gates get validated status.
  Both jobs have a 3600-second SentinelX limit; inspect partial logs
  and resume unfinished cases if the limit is reached.
- Telemetry source fix isolated from the pinned candidate:
  codex/graph-memory-observation, commit 89360ef0, draft PR #64.
  It reports null for saturated/nonpositive global free-byte deltas,
  logs raw before/after values, and rejects legacy zero readings in
  the graph-enabled allowance sweep. Windows Python sweep regressions
  8/8 passed. C++ build/tests and hosted CI are not yet run.
  No graph default, master, integration binary, or release tag changed.
