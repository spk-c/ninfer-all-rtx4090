# NInfer v0.1.4 versus llama.cpp: limited HTTP pilot (2026-11-26)

One RTX 4090, Windows, sequential runs; not a universal engine ranking. Exact NInfer package verified in [release report](../releases/rtx4090-v0.1.4-gpu-verification.md). llama.cpp 0.6.0-dev build 11617 commit 14310190c; server executable SHA-256 `715f0ea98819f03fc596e9d352e989fc0ae093056282e0d73b2d657e7afa7d02`.

The NInfer conversion record names the local Huihui IQ3_S MTP GGUF as its source text file. GGUF SHA-256 `eea0638e283433e27b0edbd409b552be467a75ecc777d91c51db554c0a644c19`; native artifact SHA-256 `a410396311ccb27048008a7ccb8a38f91099f6f6a8e10d10bd8baed7248c046a`. Related text weights, but conversion, cache layout, math and runtime may differ.

- NInfer: 8192 context and explicit KV capacity, BF16 KV, one sequence, no speculation, no prefix reuse.
- llama.cpp: 8192 context, one slot, GPU layers 99, F16 K/V, Flash Attention on, reasoning off, no prompt cache. No speculative backend requested.
- Both: same local OpenAI chat endpoint workload, one distinct warmup, three prompts twice, temperature 0, seed 123, 128 completion-token limit, thinking off, cache_prompt=false. HTTP client wall clock around the complete non-streaming request; actual usage counts. Prompts measured 27/26/28 tokens in both; zero cached tokens reported. Each completed with 128 tokens and finish_reason=length.

| Engine | Requests | Output tokens | Sum wall s | Aggregate output tok/s | Mean wall latency |
|---|---:|---:|---:|---:|---:|
| NInfer v0.1.4 | 6 | 768 | 11.198919 | 68.5780 | 1.8665 s |
| llama.cpp 14310190c | 6 | 768 | 12.831714 | 59.8517 | 2.1386 s |

NInfer was faster on this small pilot. **Exact output string equality was 0/6.** This is not a quality comparison or proof of token equivalence. There was no randomized/interleaved process order, no long prompts or concurrent requests, no matching quantized KV formats, and no comprehensive thermal/power control. No vLLM results: WSL is not supported under the test service's LocalSystem identity. Do not extrapolate.

## Prompts and commands

1. Write a detailed explanation of why the sky appears blue, with three examples.
2. List ten useful debugging techniques for a C++ application and explain each.
3. Explain the differences between a GPU and a CPU in a detailed technical answer.

Warmup: Explain a simple sorting algorithm in a few sentences.

```powershell
ninfer-serve.exe model.ninfer --host 127.0.0.1 --port 18085 --model-id compare --max-context 8192 --kv-capacity 8192 --max-concurrency 1 --kv-dtype bf16 --spec none --no-prefix-reuse
llama-server.exe -m source.gguf --host 127.0.0.1 --port 18086 --alias compare -c 8192 -np 1 -ngl 99 -ctk f16 -ctv f16 -fa on --reasoning off --no-cache-prompt
```

Local raw JSON retained under C:\ProgramData\SentinelX\ninfer-v014-release-proof: compare-ninfer.json SHA-256 `3771d78728c60fe4020c2716aa79981049d5a352f62c2495cf53f1133f1e8561`; compare-llama.json SHA-256 `757c9dd4ccc0f0749c4a712bd5d6dc5ff140784b037c810a2ff88370d6153a71`. Full logs alongside. Next: interleave independent restarts, longer prompts, output quality and matched KV/quantization where possible.
