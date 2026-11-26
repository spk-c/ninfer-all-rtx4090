# Distribution and community launch checklist

Current official website: https://spk-c.github.io/ninfer-all-rtx4090/

## Repository owner actions

- In GitHub About edit Website to the URL above. Suggested description: Native RTX 4090 (SM89) CUDA 13.4 local LLM inference engine. Qwen, Bonsai, GGUF-derived artifacts, DFlash2, MTP, OpenAI-compatible API and portable Windows binaries.
- Enable Discussions, and upload the project's 1200x630 social-preview.png from gh-pages via Settings → Social preview.
- Suggested 20 topics: rtx4090, llm, cuda, local-llm, local-ai, llm-inference, llm-serving, nvidia, gguf, qwen, dflash2, speculative-decoding, cuda-graphs, gpu-optimization, ada-lovelace, sm89, quantization, openai-compatible, windows, benchmarking.
- From the authenticated desktop, run `python scripts/update-hf-website-backlinks.py` for preview and append `--apply` to publish both model cards. Only README.md is edited; the script requires SPK1337, uses revision leases and read-back verification.
- Google Search Console: verify URL-prefix ownership, submit sitemap.xml and inspect canonical URLs. Bing Webmaster Tools: import verified Search Console property or verify ownership. Indexing and analytics require account access; do not assume indexed just because HTTP 200.
- No tracking script by default. Choose analytics provider, obtain site ID/token and document privacy obligations before enabling.
- Review each community's current rules before posting and submit only once, disclose project affiliation; no requests for coordinated voting.

## Reddit draft (r/LocalLLaMA, check rules before posting)

**NInfer RTX 4090: native CUDA Windows release, Qwen/Bonsai support and reproducible inference tests**

I'm maintaining an RTX 4090-focused NInfer-all derivative, now with a stable CUDA 13.4 SM89 Windows release. It's a native C++/CUDA engine, not a wrapper around llama.cpp. The exact v0.1.4 archive passed a real 27B model GPU smoke test. I've added a free model/VRAM command planner, an English/German guide and a benchmark dataset.

On one fixed synthetic short Engine workload (2048 prompt/128 decode), Huihui IQ3_S with DFlash2 K=7 reached 326.4 tok/s; this is not HTTP serving throughput or a general speed guarantee. A separate limited no-speculation HTTP pilot was 68.58 vs 59.85 output tokens/s for NInfer vs my local llama.cpp build; the outputs differed, cache formats were not identical, and this is not a quality or comprehensive speed comparison.

Website: https://spk-c.github.io/ninfer-all-rtx4090/
Source and releases: https://github.com/spk-c/ninfer-all-rtx4090
I would appreciate reproducible feedback, especially negative results and long-context measurements.

## Show HN draft

Title: Show HN: NInfer RTX 4090 – Native CUDA LLM inference on Windows

URL: https://spk-c.github.io/ninfer-all-rtx4090/

Comment: I'm the maintainer of this RTX 4090-focused project derived from NInfer-all. The stable Windows SM89 binary passed real-model smoke testing. The site includes a model catalog, a local command generator, and benchmark methodology with limitations. We're looking for independently reproducible results, not making universal speed claims.

## Awesome-list submission

Identify a currently maintained list whose rules permit inference engines and recently published projects. Read contribution instructions and check for duplicate NInfer entries. Proposed entry: [NInfer-all RTX 4090](https://github.com/spk-c/ninfer-all-rtx4090) — Native SM89 C++/CUDA LLM inference for RTX 4090, portable Windows packages, model-specific speculation and local APIs. Only open a PR if it fits the list and the maintainers' rules. 
