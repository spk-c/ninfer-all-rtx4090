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
5. The final worker restores the compiler caches, links the six portable
   programs and checks that cached compilation was actually reused.
   It builds the native test targets and runs CPU-only CTest. Tests that
   require an actual GPU/model may skip.
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
