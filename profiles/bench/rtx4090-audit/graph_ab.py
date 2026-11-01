"""Unprofiled paired CUDA Graph/eager comparison; raw output equality is checked."""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--cli", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
if a.out.exists():
    p.error("output exists")
prompt = ("Review this Python module for a concrete bug and explain briefly.\n" +
          "\n".join(f"def function_{i}(items): return items[{i} % len(items)]" for i in range(80)))
common = [str(a.cli.resolve()), str(a.model.resolve()), "--prompt", prompt,
          "--max-context", "8192", "--kv-capacity", "8192", "--max-new", "128",
          "--greedy", "--no-thinking", "--spec", "dflash2", "--draft-tokens", "5",
          "--lm-head-draft", "--kv-dtype", "rk4v4", "--gdn-state-fp16",
          "--prefill-chunk", "1024"]
rows, ratios, failures = [], [], []
for rep in range(5):
    pair = {}
    for arm in (("eager", "graph") if rep % 2 == 0 else ("graph", "eager")):
        command = common + (["--no-cuda-graph"] if arm == "eager" else [])
        telemetry_command = ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.gr,clocks.mem,memory.used,utilization.gpu",
                             "--format=csv,noheader"]
        before = subprocess.run(telemetry_command, capture_output=True, text=True, check=True).stdout.strip()
        start = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=300)
        wall = time.perf_counter() - start
        log = re.sub(r"\x1b\[[0-9;]*m", "", result.stderr)
        metric = re.search(r"decode speed\s+([\d.]+)\s*(?:tok/s|tok/s)", log)
        row = {"rep": rep, "arm": arm, "command": command, "returncode": result.returncode,
               "wall_s": wall, "stdout": result.stdout, "stderr": log,
               "decode_tokens_per_second": float(metric[1]) if metric else None}
        row["gpu_before"] = before
        row["gpu_after"] = subprocess.run(telemetry_command, capture_output=True,
                                          text=True, check=True).stdout.strip()
        rows.append(row)
        pair[arm] = row
        print(rep, arm, result.returncode, row["decode_tokens_per_second"], flush=True)
        if result.returncode or not metric:
            failures.append(f"{rep}/{arm}: failed or decode metric missing")
    identical = pair["graph"]["stdout"] == pair["eager"]["stdout"]
    if not identical:
        failures.append(f"{rep}: graph/eager stdout differs")
    if all(pair[arm]["decode_tokens_per_second"] for arm in pair):
        ratios.append(pair["graph"]["decode_tokens_per_second"] /
                      pair["eager"]["decode_tokens_per_second"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rows": rows, "paired_graph_over_eager": ratios,
        "paired_median": statistics.median(ratios) if ratios else None,
        "failures": failures}, indent=2) + "\n", encoding="utf-8")
raise SystemExit(1 if failures else 0)
