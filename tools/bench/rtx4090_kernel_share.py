"""Summarise Nsight Systems cuda_gpu_kern_sum CSV by quantized weight format.

Uses summed kernel durations, NOT GPU busy time, CPU time or end-to-end speed.
Overlapping kernels may double-count elapsed wall time. No numbers are generated
without a measured Nsight CSV. Use on RTX 4090 SM89 trace captures only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

FAMILIES = ("IQ4_XS", "IQ3_S", "IQ3_XXS")
# MSVC/CUDA demangles ggml template arguments as numeric enum casts. These are
# stored-format IDs from third_party/ggml-quants/include/ggml.h, not tile sizes.
GGML_TYPE_IDS = {"IQ4_XS": 23, "IQ3_S": 21, "IQ3_XXS": 18}
PATTERNS = {
    family: re.compile(symbol + rf"|\(\s*(?:enum\s+)?ggml_type\s*\)\s*{GGML_TYPE_IDS[family]}\b", re.I)
    for family, symbol in {"IQ4_XS": r"iq4[_-]?xs", "IQ3_S": r"iq3[_-]?s(?![a-z])",
                           "IQ3_XXS": r"iq3[_-]?xxs"}.items()
}


def analyse(rows: list[dict[str, str]], *, require_iq4: bool = False) -> dict:
    if not rows:
        raise ValueError("Nsight kernel-summary CSV has no data rows")
    timings: list[tuple[str, int, int]] = []
    for row in rows:
        name = row.get("Name", row.get("Kernel Name", "")).strip()
        if not name:
            raise ValueError("Nsight CSV row has no kernel name")
        try:
            ns = int(row["Total Time (ns)"].replace(",", "").strip())
            count = int(row.get("Instances", "1").replace(",", "").strip())
        except (KeyError, ValueError, AttributeError) as exc:
            raise ValueError("invalid Nsight total-time/instances data for " + name) from exc
        if ns < 0 or count < 1:
            raise ValueError("Nsight durations must be nonnegative and instances positive")
        timings.append((name, ns, count))
    total_ns = sum(ns for _, ns, _ in timings)
    if total_ns <= 0:
        raise ValueError("Nsight CSV has no measured kernel duration")
    groups = {}
    for family in FAMILIES:
        chosen = [(name, ns, count) for name, ns, count in timings
                  if PATTERNS[family].search(name)]
        duration = sum(ns for _, ns, _ in chosen)
        groups[family] = {
            "total_time_ns": duration,
            "share_percent": round(duration * 100.0 / total_ns, 4),
            "instances": sum(count for _, _, count in chosen),
            "kernel_rows": len(chosen),
        }
    if require_iq4 and not groups["IQ4_XS"]["kernel_rows"]:
        raise ValueError("no IQ4_XS kernels in this trace; check the model and capture range")
    largest = sorted(timings, key=lambda item: item[1], reverse=True)[:10]
    return {
        "schema_version": 1,
        "scope": "sum of Nsight kernel durations, NOT elapsed GPU wall / CPU time",
        "numeric_ggml_type_ids": GGML_TYPE_IDS,
        "total_kernel_time_ns": total_ns,
        "kernel_rows": len(timings),
        "families": groups,
        "top_kernels": [{"name": name, "total_time_ns": ns, "instances": count}
                        for name, ns, count in largest],
        "limitations": ("Nsight captures kernels only inside its configured range. "
                        "Overlap may double-count GPU elapsed wall time. "
                        "This report cannot establish a decode-speed gain."),
    }


def parse_nsys_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fp:
        # Nsight may prepend metadata/comment lines before the header.
        rows = csv.reader(fp)
        header = None
        body = []
        for row in rows:
            if not row:
                continue
            cleaned = [part.strip() for part in row]
            if header is None:
                if "Total Time (ns)" in cleaned and (
                    "Name" in cleaned or "Kernel Name" in cleaned
                ):
                    header = cleaned
                continue
            if len(row) != len(header):
                raise ValueError("malformed Nsight CSV row (column count mismatch)")
            body.append(dict(zip(header, row)))
        if header is None:
            raise ValueError("expected Nsight cuda_gpu_kern_sum columns missing")
        return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="cuda_gpu_kern_sum CSV")
    parser.add_argument("--output", type=Path, required=True, help="JSON summary")
    parser.add_argument("--require-iq4", action="store_true",
                        help="fail rather than reporting an artificial 0%% IQ4_XS share")
    args = parser.parse_args(argv)
    try:
        data = analyse(parse_nsys_csv(args.input), require_iq4=args.require_iq4)
        data["source_file"] = str(args.input)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for family in FAMILIES:
        item = data["families"][family]
        print(f"{family:8} {item['share_percent']:8.3f}% "
              f"({item['instances']} kernels, {item['kernel_rows']} rows)")
    print(data["limitations"])
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
