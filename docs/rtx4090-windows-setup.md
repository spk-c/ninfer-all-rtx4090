# Optional Windows installation and launcher

These tools are separate from the existing portable run.bat/download-model.bat and do not change Engine defaults.

PowerShell 7+ on Windows, 7-Zip, compatible NVIDIA driver, VC++ runtime:

```powershell
.\scripts\install-rtx4090.ps1
.\scripts\launch-rtx4090-gui.ps1
```

The installer selects the latest stable SM89 release, checks each archive/manifest against GitHub asset metadata and SHA-256, tests the archive and validates BUILD-INFO/version before moving it into a new versioned directory under LocalAppData. It refuses to overwrite an existing version. Models are not installed. Use -ArchiveCache to reuse a downloaded verified package and -Destination for a different install root.

The optional GUI selects an existing extracted runtime and .ninfer file, context, KV cache, speculative mode, draft count and localhost port. It defaults to no speculation and validates arguments. Backends still must be supported by the selected artifact; no VRAM fit guarantee. It starts/stops only its own server. GUI requires an interactive desktop; the argument validation mode works without one.

```powershell
.\scripts\launch-rtx4090-gui.ps1 -RuntimeDir C:\NInfer\rtx4090-v0.1.4 -ModelPath C:\models\model.ninfer -ValidateOnly
```

## Coding-agent connection

For clients that support an OpenAI-compatible provider, configure base URL `http://127.0.0.1:8080/v1` and model `local-model` (or actual model ID from GET /v1/models). Configure tool calling, reasoning and context limits according to the client and artifact. Use a placeholder API key only if the client requires one and server authentication is disabled. Never expose an unauthenticated server beyond localhost.

For more detailed per-model flags, use the [online configurator](https://spk-c.github.io/ninfer-all-rtx4090/configurator.html). 
