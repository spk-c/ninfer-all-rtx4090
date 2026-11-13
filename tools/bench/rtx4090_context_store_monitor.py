"""Sample an already-running NInfer server's Context/ReplaySSM/cache metrics.

Read only. Does not enable or clear cache tiers, modify the model, or send any
generation requests. Counter resets are reported, never hidden as speedups.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.request
from typing import Sequence


LINE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+'
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|NaN|[+-]?Inf)'
    r'(?:\s+\S+)?$'
)
KEYWORDS = (
    "cache", "context", "snapshot", "spill", "checkpoint", "kv",
    "prefix", "replay", "prefill", "host_", "disk_", "state_",
    "admission", "waiting", "paused", "request_queue", "first_token",
    "hydration", "restore", "transfer", "tokens_predicted", "prompt_tokens",
)


def parse_metrics(body: str) -> dict[str, float]:
    entries: dict[str, float] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        matched = LINE.fullmatch(line)
        if not matched:
            raise ValueError(f"unexpected Prometheus metrics row: {line[:160]}")
        name, raw = matched.groups()
        value = float(raw)
        if not math.isfinite(value):
            continue
        if name in entries:
            raise ValueError(f"duplicate Prometheus series: {name}")
        entries[name] = value
    return entries


def selected_series(entries: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in entries.items()
            if (k.startswith(("ninfer:", "ninfer_", "llamacpp:"))
                and any(word in k.lower() for word in KEYWORDS))}


def differences(start: dict[str, float], end: dict[str, float]) -> dict:
    names = set(start) & set(end)
    return {
        "series_in_both": len(names),
        "missing_at_end": sorted(set(start) - set(end)),
        "new_at_end": sorted(set(end) - set(start)),
        "negative_changes_possible_reset": sorted(
            key for key in names if end[key] < start[key]),
        "observed_changes": {key: end[key] - start[key]
                             for key in sorted(names) if end[key] >= start[key]
                             and end[key] != start[key]},
        "warning": "Differences include gauges; a counter reset invalidates rates. "
                   "This is not a performance benchmark.",
    }


def sample(url: str, api_key: str | None, timeout: float) -> dict[str, float]:
    request = urllib.request.Request(url.rstrip("/") + "/metrics")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"/metrics returned HTTP {response.status}")
        body = response.read(8 * 1024 * 1024 + 1)
        if len(body) > 8 * 1024 * 1024:
            raise ValueError("/metrics response too large")
    return selected_series(parse_metrics(body.decode("utf-8")))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--interval-seconds", type=float, default=10)
    p.add_argument("--timeout", type=float, default=10)
    p.add_argument("--api-key-env", default="NINFER_API_KEY")
    opts = p.parse_args(argv)
    if not 1 <= opts.samples <= 10000:
        p.error("samples must be 1..10000")
    if not math.isfinite(opts.interval_seconds) or opts.interval_seconds < 0:
        p.error("interval must be finite and nonnegative")
    if not math.isfinite(opts.timeout) or opts.timeout <= 0:
        p.error("timeout must be finite and positive")
    if opts.output.exists():
        p.error(f"refusing to overwrite {opts.output}")
    # Require loopback unless the operator deliberately configures an HTTPS URL.
    if not re.match(r"^https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$", opts.url.rstrip("/")):
        p.error("only a local loopback server is supported (no remote token exposure)")
    key = os.environ.get(opts.api_key_env)
    snapshots = []
    for i in range(opts.samples):
        if i:
            time.sleep(opts.interval_seconds)
        measured = sample(opts.url, key, opts.timeout)
        if not measured:
            p.error("no NInfer cache-related metrics found; verify the endpoint")
        snapshots.append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "selected_metrics": measured,
        })
        print(f"sample {i+1}/{opts.samples}: {len(measured)} cache-related series",
              flush=True)
    result = {
        "schema_version": 1, "url": opts.url, "samples": snapshots,
        "difference": differences(snapshots[0]["selected_metrics"],
                                  snapshots[-1]["selected_metrics"]),
        "privacy": "no request text or API keys are logged; read-only metrics",
        "limitations": "Metrics are server-wide; workload attribution requires an otherwise idle server.",
    }
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
