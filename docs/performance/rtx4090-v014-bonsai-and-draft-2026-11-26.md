# RTX 4090 v0.1.4: Bonsai 2.5 and short draft-width tests

**2026-11-26, exact downloaded v0.1.4 benchmark executable**, Windows x64 on RTX 4090. See [release proof](../releases/rtx4090-v0.1.4-gpu-verification.md). Build CUDA 13.4; the Engine JSON reports CUDA runtime and driver API 13.5 (driver 621.40). Source a6b1138b528b476286bc6b11f19ba30df196b6ed.

## Results

Combined `pp2048+tg128` row, 2,048 input tokens / 128 timed decode tokens, explicit 32,768 context, rk4v4, GDN FP16, 2 measured repetitions and one warmup, standard token ID corpus. Decode stddev is the report's within-run value, not confidence across independent process repeats.

| Model | Backend | Prefill tok/s | Decode tok/s ± SD | Accepted / drafted | Weights + runtime reservation GiB | N-gram width |
|---|---|---:|---:|---:|---:|---:|
| Huihui IQ3_S | none | 3874.13 | 70.38 ± 0.13 | — | 11.88 | 0 |
| Huihui IQ3_S | DFlash2 3 | 3805.45 | 201.93 ± 0.27 | 93.07% | 14.78 | 0 |
| Huihui IQ3_S | DFlash2 5 | 3766.16 | 257.13 ± 0.07 | 85.83% | 14.78 | 0 |
| Huihui IQ3_S | DFlash2 7 | 3729.58 | 326.41 ± 0.32 | 88.00% | 14.79 | 0 |
| Huihui IQ3_S | DFlash2 9 | 3819.98 | 269.33 ± 0.06 | 74.50% | 14.79 | 0 |
| Huihui IQ3_S | MTP 3 | 3799.12 | 200.31 ± 0.10 | 89.42% | 12.67 | 0 |
| Huihui IQ3_S | MTP 5 | 3766.82 | 242.59 ± 0.24 | 85.83% | 12.68 | 0 |
| Bonsai 2.5 PTQ1_0 | MTP 3 | 4635.36 | 291.69 ± 0.12 | 66.43% | 7.13 | 15 |
| Bonsai 2.5 PQ2_0 | MTP 3 | 6295.90 | 323.88 ± 0.72 | 63.51% | 8.21 | 15 |

- Huihui: no n-gram copy proposals in all seven arms. The tested short fixed corpus favors DFlash2 K=7; K=9 is slower and has lower acceptance. The initial command timed out after K=7 and K=9 was repeated separately; MTP K=3 and K=5 were then tested. No MTP K=7/9 result is claimed.
- Bonsai 2.5: MTP 3 with default n-gram width 15; do not compare its acceptance directly with the no-copy Huihui sweep. Different artifacts have different weights, quality and accepted drafts. PQ2_0 has more weight storage and faster decode on this test; PTQ1_0 uses less memory.
- Engine allocation sums exclude desktop, CUDA allocator and other overhead. Main KV planner is not the complete runtime reservation.
- These are short sequential synthetic corpus tests, not a standardized HTTP workload, quality/perplexity test, long-context validation, a controlled interleaved comparison against earlier releases, or a comparison to llama.cpp or vLLM.

## Reproduce

Use the exact published package with its manifest, compatible model files and the repository corpus at the release source. Corpus SHA-256 `054f7f732f6f75c7d8cc16f8d76a05aa0af91d7c7533ed2dab0afb41884b988d`.

```powershell
.\ninfer_bench.exe --weights C:\models\model.ninfer --corpus C:\repo\bench\fixtures\bench_corpus.ids -p 2048 -n 128 -pg '2048,128' -r 2 --warmup 1 --max-ctx 32768 --kv-dtype rk4v4 --gdn-state-fp16 --ngram-draft-tokens 0 --spec dflash2 --draft-tokens 7 -o json --output-file result.json
```

For none, omit --spec and --draft-tokens. For MTP, select --spec mtp. For Bonsai use MTP3 and the default n-gram width 15. Verify GPU idle before each test.

[Machine-readable summary](rtx4090-v014-bonsai-and-draft-2026-11-26.json) contains full-precision values, raw report hashes and explicit scope. The original complete Engine JSON and logs are retained under C:\ProgramData\SentinelX\ninfer-v014-release-proof on the test host. The summary is not the complete raw dataset.
