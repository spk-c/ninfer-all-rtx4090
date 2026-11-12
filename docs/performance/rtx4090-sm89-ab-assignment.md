# RTX 4090 / SM89 — fixed-master benchmark and optimization assignment

> Tracking issue: [#36](https://github.com/spk-c/ninfer-all-rtx4090/issues/36). This is an executable **qualification plan**, not a completed benchmark or a kernel-speed claim. The planning branch must not change production kernel defaults.

## 1. Immutable comparison contract

- Arm A = `spk-c/ninfer-all-rtx4090` master commit `f879a2db06c15133260551a104feaf03886cc620` (the master HEAD when this assignment was created). Do **not** silently follow a moving `master` branch; write down any new reference SHA and issue a new campaign if it moves.
- Arm B = the separate feature/optimization commit being tested. Compare identical `.ninfer` model bytes (SHA-256), companion DFlash2 weights, tokenizer, prompts, seed, sampling, quantization and context where the test dimension does not vary.
- Both executables need independent complete Windows runtime/DLL trees, native CUDA 13.4.3 / `sm_89`, same MSVC/vcpkg inputs, same driver and clock/power conditions. Do not copy an EXE from master into the candidate's `apps` folder.
- Target machine: 24 GiB GeForce RTX 4090. Close other GPU workloads; preserve normal power limit and record nvidia-smi temperatures, graphics/memory clocks, driver, CUDA version and VRAM before/after each case.
- Preflight: each binary's `--version` should identify its recorded build. Record Git SHAs of both source trees, hashes of EXEs and relevant dependent libraries, model SHA-256, fixtures and parsed prompt-token counts.

## 2. Generate the pinned 54-case workload plan

From the repository root, run the following example with the real paths and the **actual 40-hex SHA** of Arm B:

```powershell
python -m tools.bench.plan_rtx4090_sm89 `
  --master-exe 'C:\ninfer-master\build-rtx4090\apps\ninfer.exe' `
  --candidate-exe 'C:\ninfer-candidate\build-rtx4090\apps\ninfer.exe' `
  --model 'C:\ninfer-all\models\Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer' `
  --fixtures-dir 'C:\ninfer-fixtures\rtx4090-sm89' `
  --candidate-sha 0123456789abcdef0123456789abcdef01234567 `
  --repetitions 5 --out '.\profiles\bench\sm89-qual-001' --strict
```

The candidate SHA in this example is **a placeholder**, not a real revision. If the Windows build outputs to `apps\\Release`, use that location for both arms and verify each independently. The planner's `--strict` option verifies that the model, 18 input fixtures and gold-answer manifest exist, then freezes SHA-256 hashes of **all input files** in the plan. A plan without `--strict` is only a **NOT_RUN** matrix suitable for review, never valid GPU evidence. No GPU performance numbers are generated.

Artifacts: `manifest.json`, `cases.json`, `fixture-requirements.json`, `ab-commands.json`, `nsight-commands.json`. The 54 cases comprise 12 DFlash2 tests, 12 prefill tests, and 30 KV/retrieval tests. At five repetitions with two independently built executables this is **540 process runs**, plus kernel profiling. The planner emits commands for the existing `tools/bench/run_interleaved_ab.py`; it does not execute them or capture their raw outputs. That execution and reporting are explicit follow-up work in issue #36.


## 2A. Reproducible fixture generation

Generate the requested 18 JSON chat fixtures **locally** from a clean copy of the frozen master; never commit multi-megabyte generated prompt bodies to this repository. The fixture generator only uses distinct source-code ranges; it rejects corpus tiling and records SHA-256 hashes and a 15-entry independently checkable gold-answer map.

```powershell
# Checkout source A at the exact fixed master SHA.
git worktree add C:\ninfer-master-snapshot f879a2db06c15133260551a104feaf03886cc620

python -m tools.bench.make_rtx4090_sm89_fixtures `
  --source-master 'C:\ninfer-master-snapshot' `
  --out 'C:\ninfer-fixtures\rtx4090-sm89' `
  --chars-per-token 3.0
```

The estimated characters/token ratio is **not** a real tokenizer measurement. Tune it and recreate the entire fixture directory if necessary; never silently accept short inputs. The evidence runner rejects KV cases under **90%** of the selected context target or where the input plus 128 decode tokens would overrun the configured context. A 128K flag alone never qualifies a 128K result.

## 2B. Execute and preserve paired evidence on a free 4090

After the baseline and candidate have each been built at their own **recorded commits** under native Windows CUDA 13.4.3 / SM89, create the strict plan with a real candidate SHA (from `git rev-parse HEAD`). Then run:

```powershell
python -m tools.bench.run_rtx4090_sm89_evidence `
  --plan-dir '.\profiles\bench\sm89-qual-001' `
  --master-source 'C:\ninfer-master-snapshot' `
  --candidate-source 'C:\ninfer-candidate' `
  --gold-answers 'C:\ninfer-fixtures\rtx4090-sm89\gold-answers.json' `
  --repetitions 5 `
  --max-idle-vram-mib 2048 `
  --out '.\profiles\bench\sm89-evidence-001'
```

The collector verifies the **actual** master and candidate Git HEAD hashes, the model/fixture/gold SHA-256 values frozen in the plan, runtime paths, CUDA compute capability **8.9**, GPU idle state and all 54 checked-in case definitions. It alternates AB/BA per case and repetition, samples whole-card GPU memory, records the CLI's actual token counts and draft acceptance, and saves all 540 stdout/stderr logs and sample metrics to `samples.csv`. GPU-memory samples are from `nvidia-smi` and represent **total card occupancy**, not a per-process allocation profile; the CLI separately reports its workspace peak. A `--max-cases 1` smoke run is allowed but is marked **PARTIAL** and cannot qualify any release.

**Caution:** the CLI's wall-clock elapsed time includes model startup and is **not server TTFT**. Use the existing `tools/bench/run_serve_ttft.py` framework for TTFT follow-up (separately matched configurations); never mislabel the CLI startup time.

## 2C. Generate the paired report

```powershell
python -m tools.bench.report_rtx4090_sm89_evidence `
  --evidence-dir '.\profiles\bench\sm89-evidence-001' `
  --out '.\profiles\bench\sm89-report-001'
```

This emits `report.json` and `summary.md`: median paired B/A, seeded bootstrap 95% interval, complete-pair count, negative/zero-gain pairs, divergent answer hashes, peak **whole-card** VRAM and explicit failures/OOM. The report **never automatically promotes kernel routes**. Review coding answer correctness and attach actual Nsight profiles, precision/perplexity evidence, release CI and long-context gold accuracy before proposing an opt-in implementation. A generated plan, completed report or microbenchmark alone is not sufficient.


## 3. Prepare truthful prompt fixtures before measurement

- Required JSON message files `coding-short.json`, `coding-long.json`, `coding-edit.json` and `kv-32768-seed1.json` through `kv-131072-seed5.json` (five distinct seeds per context depth, 15 files). Each JSON must be a supported `--messages` array with realistic coding context, not the tiled synthetic benchmark corpus.
- **Short/long coding:** Keep each byte-for-byte identical across all DFlash2 K-values and prefill candidates; make long prompts contain actual code review context and record the prompt-token count printed by NInfer. Control the requested output token count and check for early termination.
- **KV long context:** A flag of `--max-context 131072` alone does not measure 128K. Generate and tokenize genuine near-limit fixtures, at least 90% of each target length and with sufficient space for 128 completion tokens. Place unique canary identifiers, code definitions and exact-answer questions early, centrally and near the end. Maintain a separate gold-answer manifest; do not infer from the prompt string length.
- Five fixed seeds × 32K/64K/128K; ask deterministic recall/symbol-resolution questions, check gold answers and complete token accounting, and preserve failures/unsupported contexts. Models that cannot admit the full depth on a 24 GiB 4090 must report OOM or unsupported with measured VRAM, not silently downsize.

## 4. Required A/B matrices

| Family | Exact matrix | Hold constant | Record |
|---|---|---|---|
| DFlash2 | K=3/5/7/9 × short/long/edit coding (12) | Model, `rk4v4`, chunk 1024, `--ngram-draft-tokens 0`, FP16 GDN, greedy/no-thinking | Decode output tok/s, draft acceptance %, tokens per round, generated tokens, answer hash, TTFT, peak VRAM |
| Prefill | cuBLAS off/on × chunk 1024/2048/4096 × short/long coding (12) | Model, `rk4v4`, no speculation, sampler, prompt | Prefill tok/s, latency/TTFT, device workspace and VRAM, output drift, quality |
| KV/context | `rk4v4` vs `rk8v4` × 32K/64K/128K × five gold-answer seeds (30) | Model, no speculation, chunk 1024, sampler, actual corpus and prompt length | Peak VRAM, spill/transfer, prefill/decode, gold-answer exact accuracy, perplexity/logit drift |

All rows must be run once for Arm A and once for Arm B within every repeated pair. Alternate AB/BA, use ≥5 valid paired repetitions plus warmups, and report **median paired B/A** (not ratio of overall medians). The dedicated SM89 collector/reporter supplies raw CLI stdout/stderr, answer digests, periodic whole-card VRAM samples and paired confidence intervals. Power/clock telemetry, HTTP TTFT, exhaustive accuracy/perplexity and Nsight qualification remain separate required measurements.

## 5. SM89 kernel profiling — three microbenchmark programs

The planner emits sample Nsight Systems and Nsight Compute command arrays for:

- `ninfer_q4_linear_swiglu_schedule_bench` — Q4 Small-T, token widths 1/2/4/8/16/32.
- `ninfer_q5_linear_add_schedule_bench` — Q5 Small-T, token widths 1/2/4/8/16/32.
- `ninfer_a8_prefill_schedule_bench` — W4A8 prefill, tokens 128/512/1024, including panel layout.

Run `ninfer_benches --list` first and confirm the program names and binary path in each separate build. Nsight Systems: capture CUDA kernel launch, Graph nodes, NVTX, memcpy, synchronization and host-side launch gaps. Nsight Compute: target the hot **actual SM89 kernel symbol** found in Nsight Systems; record registers/thread, theoretical and achieved occupancy, shared-memory use, local-memory spills, L1/L2, DRAM throughput/bytes, tensor-core utilization and latency. Resolve available metric names from the installed NCU version (`ncu --query-metrics`); do not paste Ampere/Blackwell ceiling constants into Ada reports.

Collect `nsys` and `ncu` artifacts, export kernel-level CSV, annotate shape/kernel/route, and compare cold/warm conditions separately. Existing `scripts/measure-rtx4090-regressions.ps1`, `docs/rtx4090.md`, `bench/README.md`, and `tools/bench/rtx4090_kernel_share.py` describe related evidence. Never infer an end-to-end improvement from a microbenchmark alone.

## 6. Evidence and promotion gate

Every compared case must retain raw stdout/stderr, actual generated tokens, exact input token count, answer hash, elapsed time, TTFT, decode/prefill rates, hardware telemetry, peak allocated/resident VRAM, code/model hashes, and a status in `{passed, failed, OOM, unsupported, not_run, quality_regression}`. Produce `samples.csv`, a case-level JSON and `summary.md` with paired median percent, distribution, ≥5 complete pairs and bootstrap confidence interval. Do not average incomplete cases with successful ones. Split results by context, concurrency and code workload.

Only propose a kernel or routing change in a *subsequent opt-in implementation PR* when:

1. The same-artifact RTX 4090 end-to-end paired median gain is at least **+3%**, with the uncertainty interval's lower bound above 0, across an eligible workload.
2. No protected benchmark regresses more than **2%**; when close to noise floor, increase repetitions before claiming a win.
3. Unchanged-precision paths pass numerical equality/regression checks. For the differing KV dtypes and cuBLAS route, attach gold-answer accuracy, perceptual/code validity and perplexity/logit drift assessments instead of claiming exact identical text.
4. Long-context 32K/64K/128K accuracy passes the five-seed gold-answer suite. OOM/unsupported is explicit and blocks any full-context marketing claim.
5. Windows release CI, native CUDA 13.4.3 / SM89 compile, CPU smoke tests, relevant GPU correctness and sanitizer checks all succeed.

## 7. Release and branch protection

Do not modify `.github/workflows/native-windows.yml`, `.github/workflows/native-release.yml`, `scripts/publish-rtx4090-windows-release.ps1` or the CUDA 13.4.3 SM89 four-shard/checksum/release outputs in this planning PR. Keep every experimental Q4/Q5/W4A8 kernel route and fast-math variant **disabled by default** until explicitly qualified. Do not auto-merge PR #35 into this branch; it contains separate unqualified optimizations and is not the baseline.

**Completion definition:** Issue #36 can close only after matched GPU run evidence, correct long-context fixtures, quality checks, reviewed kernel traces, the comparison report and CI logs are linked. This initial planning PR only creates the structured task, enumeration and safety checks; it is not a completed optimization.
