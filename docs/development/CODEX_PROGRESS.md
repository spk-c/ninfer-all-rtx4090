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
