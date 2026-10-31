
cd C:\coding\ninfer-all

New-Item -ItemType Directory -Force ".\logs" | Out-Null

& ".\build-rtx4090-coding\apps\ninfer-serve.exe" `
  "C:\ninfer-all\models\Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer" `
  --model-id qwen3.8-27b `
  --host 127.0.0.1 `
  --port 8080 `
  --device 0 `
  --max-context 176128 `
  --kv-capacity auto `
  --kv-headroom-mib 1024 `
  --kv-dtype rk4v4 `
  --gdn-state-fp16 `
  --max-concurrency 2 `
  --pending-timeout-ms 600000 `
  --spec dflash2 `
  --draft-tokens 5 `
  --ngram-draft-tokens 15 `
  --ngram-min-match 12 `
  --prefill-chunk 1024 `
  --use-original-prefix-caching `
  --kv-lease-growth `
  --derive-session-keys `
  --auto-long-anchors `
  --host-cache-mib 16384 `
  --max-private-continuations 4 `
--device-profile auto `
--device-profile-path "C:\coding\ninfer-all\build-rtx4090-coding\profiles\rtx4090-cuda134-sm89.json" `
  --structured-output `
  --vision `
  --vision-residency overlay `
  --default-reasoning-effort xhigh `
  --recover-invariant-failures `
  --log-stats-interval-ms 15000 `
  --request-log-jsonl ".\logs\req-huihui-K5-legacy-optimized.jsonl" `
  --request-log-max-mib 200 `
  --request-log-keep 3
