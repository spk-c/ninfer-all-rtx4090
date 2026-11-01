# Built-in NInfer Monitor

The native read-only dashboard is compiled into every ninfer-serve build, including
Windows builds without NINFER_WEBUI_DIR. No separate Python process, downloaded
assets, CDN or external web service is needed.

## Open

With the default HTTP port, open **http://127.0.0.1:8080/monitor**.
If you started NInfer with `--port 8081`, open **http://127.0.0.1:8081/monitor**
instead. The dashboard always polls the server at the same origin.

The page reads the existing GET /stats and GET /metrics endpoints once every
three seconds. It shows a time-windowed decode/prefill rate (differences of
monotonic token counters), active/waiting requests, device KV occupancy,
draft acceptance, uptime and aggregate request activity. In router mode,
enter the exact model ID in the Model input to inspect a particular model.

## NVIDIA GPU sensors

GET /monitor/gpu?index=0 provides optional NVML readings, including
GPU utilization, temperature, power, VRAM used and capacity. Choose an index
from 0 through 7 in the UI. NVML is loaded dynamically; no NVIDIA import
library, SDK or new build-time dependency is required. Unsupported or
unavailable sensors are displayed as a dash, not substituted with fabricated
numbers. This index is the NVIDIA NVML index; if the model is on a
different GPU, select that index explicitly.

On Linux, the build links CMAKE_DL_LIBS for the dynamic loader. On Windows,
it loads nvml.dll at runtime. Queries use the driver management API and do
not access the inference CUDA context.

## Authentication and privacy

- GET /monitor and /monitor/ are public **static HTML documents**, like
  optional WebUI assets. No model statistics, secrets or prompts are embedded.
- GET /stats, GET /metrics and GET /monitor/gpu retain normal API-key
  authentication. With `--api-key` configured, enter it into the page's
  password field. The value stays in tab memory and is not written to
  localStorage, cookies or URL query parameters.
- The HTML response uses a restrictive Content-Security-Policy; data is
  fetched only from the page's own origin.
- Keep `--host 127.0.0.1` for local-only access, unless remote access
  was intentionally configured.

## Measurement limitations

- Decode/prefill rates are differences of **server-wide completed/committed
  token counters** over the most recent polling interval. The first reading
  has no prior sample, so rates start with an empty value.
- Draft acceptance comes from totals recorded for completed requests
  since server startup; it is not an instantaneous DFlash2-only rate.
- KV occupancy is not the total submitted chat context length.
- Request activity is an aggregate of successive poll samples, **not
  a per-request event log**.
- Samples are stored only in the open browser tab, to a maximum of 120.
  The tab stops polling while hidden.

For detailed per-request JSONL history, CPU/RAM, sensor overrides and
longer history, the existing **tools/monitor/monitor.py** sidecar remains
available; see tools/monitor/README.md.
