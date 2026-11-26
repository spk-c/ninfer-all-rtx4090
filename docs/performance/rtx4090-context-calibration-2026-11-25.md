# RTX 4090 transfer calibration (2026-11-25)

The new standalone tool builds without the entire model runtime:

    cmake -S . -B build-sm89 -DCMAKE_CUDA_ARCHITECTURES=89 -DNINFER_BUILD_BENCHMARKS=ON
    cmake --build build-sm89 --target ninfer_context_transfer_bench --config Release
    ninfer_context_transfer_bench --suite transfer --transfer-warmup 5 --transfer-reps 31 --json transfer.json --preset-out context-cost.json

Use the executable path appropriate to the selected CMake generator. The standalone executable uses the same production paged-KV transfer routines and fitting/reporting code as the complete context-cost benchmark. It requires an actual CUDA device but no model artifact. The existing complete benchmark remains necessary for text/Vision prefill calibration.

## Local Windows SM89 evidence

RTX 4090, CUDA 13.4, driver 621.24. 83 distinct cases, including fragmented page transfers and contiguous D2D; repeated with 31 measurements per case and five warmups. The repeated fit was rejected by the unchanged 35% p95 training/validation threshold.

| Direction | Training p95 | Held-out p95 | Accepted |
|---|---:|---:|---|
| D2D | 45.69% | 30.76% | no |
| D2H | 47.46% | 26.97% | no |
| H2D | 21.11% | 23.11% | yes |

All three held-out direction fits had zero ordering failures. These are regression-fit errors, not transfer throughput or model speed. No preset was generated or activated. The first run (9 repetitions) also rejected the overall calibration. Raw JSON reports are retained on the test host; this is a diagnostic result, not a qualified replacement for compiled defaults. Prefill cost, startup speculative calibration and real-model performance remain unverified.
