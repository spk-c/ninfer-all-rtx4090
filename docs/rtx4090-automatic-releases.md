# Automated Windows RTX 4090 releases

The **RTX 4090 automatic Windows release** workflow builds native Windows CUDA 13.4 / SM89
from a new immutable release tag. It runs CTest, verifies the portable package
and performs a real DFlash2 model inference test before publishing.

## One-time setup

1. As a repository administrator, open **Settings > Actions > Runners > New
   self-hosted runner**. Install GitHub's Windows x64 runner on a trusted
   RTX 4090 machine. Assign it the custom **rtx4090** label, along with the
   standard self-hosted, Windows and X64 labels. Obtain the temporary
   registration token only from GitHub's UI; never publish it.
2. Install **VS 2022 Build Tools with C++ and Windows SDK**, **CUDA 13.4**,
   **CMake**, **Ninja**, **7-Zip**, **PowerShell 7**, **Git**, **GitHub CLI**,
   **Python 3.11** and vcpkg. All command-line tools must be accessible to
   the runner account. Set the VCPKG_ROOT environment variable and grant
   its account write permissions for C:\ninfer-ci\build-rtx4090-release.
3. Keep a valid Qwen3.8 NInfer model on that machine, by default at
   C:\ninfer-all\models\Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer.
   Otherwise set the repository Actions variable **NINFER_RELEASE_MODEL** to
   its absolute location. Model weights are never uploaded.
4. Keep the runner dedicated to *trusted release jobs*, not arbitrary public
   pull requests. A runner on a personal desktop has elevated security risk:
   use a dedicated non-administrator account. Protect release tags or restrict
   who can create them. Do not provide registration tokens or secrets in chats.
5. The GPU must be idle, with at least 19000 MiB free for the smoke test.
   The workflow does not kill running inference servers. If insufficient
   memory is available, rerun the failed job after freeing the GPU.

## How to publish

After merging and validating a source revision, run:

    git fetch origin master
    git switch master
    git pull --ff-only origin master
    git tag rtx4090-v0.1.2
    git push origin rtx4090-v0.1.2

Never move an existing release tag. A tag push starts the Windows release
workflow. It pins the tagged source, reuses a build directory outside the
working tree, explicitly selects Python 3.11, compiles the native SM89
release, runs CTest and packages all six portable executables. It checks
the CUDA device images, package metadata, archive and SHA256 manifest, then
runs real RTX 4090 DFlash2 inference with sufficient thinking budget.

The workflow uploads to a *prerelease* using the built-in Actions token,
verifies remote asset names and sizes, and only then promotes to stable/latest.
It refuses to overwrite an existing release. Manual workflow dispatch allows
testing an existing tag without publishing (publish=false), or publishing
when all gates pass (publish=true).

## What is fixed in the fork

The general CI workflow uses the latest published release *tag* as its
immutable comparison base rather than depending on an inaccessible upstream
GHCR image. If the base is missing or cannot be fetched, all check scopes
are enabled as a safe bootstrap. PR checks use their merge parent.

The inherited universal native-release and HF cache-retention workflows
use the upstream project's private build-cache bucket, and are skipped in
this fork. The new dedicated RTX 4090 workflow does not require that bucket.

Existing stable version rtx4090-v0.1.1 is unchanged. Release qualification
covers Windows x64, one RTX 4090, native SM89 and CUDA 13.4; it does not
certify Linux, multi-GPU, other models or all optional model-dependent tests.
