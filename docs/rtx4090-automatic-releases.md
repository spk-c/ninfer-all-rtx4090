# RTX 4090 Windows releases: GitHub-hosted multi-shard build

**No self-hosted runner is required to build and publish Windows binaries.**
For a new immutable version tag, GitHub Actions starts four windows-2022
workers in parallel. Each compiles a deterministic subset of the CUDA 13.4 /
native SM89 object graph using build-object-shard.py.

## What happens on a new tag

1. Verify the source revision is reachable from master and run the existing
   Linux/Windows release-tooling checks.
2. Four hosted Windows workers install the same pinned CUDA 13.4.3, MSVC
   2022, vcpkg baseline, Python 3.11, CMake/Ninja and ccache. Their object
   lists are disjoint and follow stable SHA256 path ownership.
3. Each exports a ZIP archive containing compiled-object ownership metadata,
   compiler cache entries and SHA256 hashes. Data moves via official
   GitHub Actions artifacts, NOT the upstream private Hugging Face cache.
4. The final Windows worker checks all four artifact identities, full Ninja
   object ownership, cache SHA256 and path safety. Missing/corrupt or stale
   artifacts cause a hard failure.
   If two shards export the same ccache pathname with different **valid**
   SHA256 payloads, the merge discards that cache key from every shard.
   It never selects an arbitrary compiler result; the final compiler must
   rebuild that key. All archive entries are verified, even discarded
   collisions and identical duplicates. The minimum cache-hit guard remains
   mandatory, so excessive collisions cause a failure rather than an
   unexpectedly expensive cold CUDA build.
5. The final worker restores the compiler caches, links the six portable
   programs and checks that cached compilation was actually reused.
   It builds the native test targets and runs CPU-only CTest. Tests that
   require an actual GPU/model may skip.
   Test staging includes the vcpkg DLL closure, `ninfer_kernels.dll` and
   both cuBLAS DLLs. CUDA 13 stores the latter in `bin/x64`; adding only
   the Toolkit's `bin` to `PATH` is insufficient. Missing DLLs produce
   Windows `0xC0000135` before a test can report that no GPU is present.
6. It packages the release, checks SM89 device images, portable startup,
   archive integrity and SHA256, and publishes verified files to Actions.
7. A separate publish job creates only a GPU-UNVERIFIED GitHub prerelease.
   It refuses to overwrite an existing tag or release.

The initial matrix uses FOUR workers, rather than eight, because installing
CUDA, vcpkg and uploading caches costs time/disk on each worker. After
a fully successful real run, use its logs to decide whether eight shards
actually save time.

## Start a new release

After merging/test-qualifying a source commit on master, create and push
a NEW version tag:

    git fetch origin master
    git switch master
    git pull --ff-only origin master
    git tag rtx4090-v0.1.2
    git push origin rtx4090-v0.1.2

The tag push automatically starts the hosted build. Manual dispatch can
run an existing tag without publishing by leaving the publish checkbox
unchecked. For an initial build-only validation without pushing a new tag,
set the version label to an unused next version (for example rtx4090-v0.1.2),
test_ref to master, and publish to false. It builds that source revision but
does not create or alter any release. Tag and release assets must remain
immutable.

## GPU qualification remains separate

GitHub-hosted Windows runners have NO RTX 4090. Their checks cannot
prove that a quantized Qwen model performs correct real GPU inference.

Download the prerelease archive and SHA256 file, verify the hash, and
unpack on an RTX 4090 computer. The repository script
scripts/test-rtx4090-release-smoke.ps1 runs the packaged server using the
real Qwen3.8 model and DFlash2. It requires at least 19 GiB free VRAM,
does not stop ongoing inference processes, and permits 256 output tokens
so the reasoning phase can complete.

Only after GPU smoke and appropriate numerical/cache regressions pass
should a maintainer explicitly promote the prerelease to Stable/Latest.
The cloud workflow never silently certifies untested GPU behavior.

This workflow qualifies Windows x64, CUDA 13.4, native SM89 and CPU
tests, not Linux, multi-GPU or optional model-dependent skipped tests.

Existing stable rtx4090-v0.1.1 and its assets remain unchanged.

## Local validation, 2026-11-03

The runtime-DLL correction and four missing LoRA/control-vector forward
declarations were checked on an owner's Windows RTX 4090 (driver 617.42,
CUDA 13.4.92, MSVC 14.44, Python 3.11.9). This is a local review build of
`237cdff9` plus uncommitted fixes, **not a published or hosted-CI-qualified
rtx4090-v0.1.2**. Its package records `modified_sources: true` and the upstream
version `2026-10-28.1`.

- Full native SM89 build with shared runtime, kernel DLL and archive PTX
  pruning enabled: passed. SFU, BF16x8 SiLU and IQ3_S loop experiments were OFF.
- Test-bundle startup with only Windows system directories on PATH: passed.
- Full CTest, `--parallel 1`: **342 passed, 32 skipped, zero failed**;
  374 tests, 3614.61 seconds. Skips include missing model artifacts and
  unsupported hardware; they are not passes.
- Intended Windows release-tooling suites and change-scope checks:
  **26 passed, six skipped**. The new loader regression failed with
  `0xC0000135` before the correction and passed afterwards; both CUDA DLL
  directory layouts and missing direct/transitive imports are covered.
- Existing `Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer`:
  DFlash2 prefill forks/split appends, exact hybrid restore, persistence and
  protocol turns passed. Cache checks used rk4v4 and FP16 GDN state.
- Packaged SM89 images, PE image bounds, all six launchers' clean-PATH
  help/version, 7-Zip integrity and SHA256: passed. The actual packaged server
  completed the real-model DFlash2 HTTP smoke.
- Packaged public Engine: pp512, pp2048, tg128 and pp2048+tg128, three measured
  repetitions each in plain and DFlash2 modes, all output budgets met.
  Context 4096, chunk 1024, rk4v4, FP32 GDN state, CUDA Graphs and the frozen
  corpus; DFlash2 used five draft tokens and its default 15-token n-gram
  proposals (plain mode disabled them). These are functional checks, not a
  matched upstream/fork performance comparison.

The first packaging attempt lacked `dumpbin` in the local shell; importing
the VS 2022 environment fixed that prerequisite, and only packaging and later
stages were rerun. An initial overbroad release-tools invocation on Windows
had 111 passes, 18 skips, 11 failures and 13 errors from Linux/environment
assumptions; it does not qualify the Linux tools. Native compilation retained
warnings 177-D/445-D/550-D and C4244/C4267 in untouched code.

The cuBLAS pair adds about 523 MiB per build output directory requiring it;
the release packager already included these DLLs. RAM/VRAM savings and an
end-to-end speedup are not established. The separate
[IQ4_XS register-cap screen](performance/rtx4090-huihui-iq3s-dflash2.md#2026-11-03-iq4_xs-register-cap-screen)
retains favorable and adverse results and rejects the candidate.

Local logs and reports are under `profiles/bench/rtx4090-release-20261103/`;
the review package is
`dist/ninfer-windows-x64-cuda13.4-sm89-237cdff94e1a-dirty/`, with its archive
and checksum beside it. Hosted validation and qualification of any subsequently
built release archive remain separate. This run does not qualify other model
artifacts, GPUs, multi-GPU inference or Linux.
