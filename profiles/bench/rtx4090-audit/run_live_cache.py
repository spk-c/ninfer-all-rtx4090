"""Start only audit-owned local servers and preserve both cache arms' commands/logs."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--server", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
statuses = []
for mode in ("hybrid", "disabled"):
    destination = a.out / f"cache-{mode}.json"
    if destination.exists():
        p.error(f"existing measurement: {destination}")
    command = [str(a.server.resolve()), str(a.model.resolve()), "--host", "127.0.0.1",
        "--port", "18989", "--model-id", "audit", "--max-context", "32768",
        "--kv-capacity", "32768", "--kv-dtype", "rk4v4", "--gdn-state-fp16",
        "--prefill-chunk", "1024", "--spec", "dflash2", "--draft-tokens", "5",
        "--lm-head-draft", "--max-concurrency", "2", "--max-pending-requests", "1",
        "--pending-timeout-ms", "2000", "--no-thinking", "--log-stats-interval-ms", "0",
        "--request-log-jsonl", str((a.out / f"requests-{mode}.jsonl").resolve())]
    command += (["--host-cache-mib", "256", "--use-alt-prefix-caching"] if mode == "hybrid"
                else ["--no-prefix-reuse"])
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    log = (a.out / f"server-{mode}.log").open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                               startupinfo=startup)
    status = {"mode": mode, "server_command": command, "pid": process.pid}
    statuses.append(status)
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited: {process.returncode}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:18989/health", timeout=1) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.25)
        else:
            raise TimeoutError("server did not become healthy")
        client = [sys.executable, "-m", "tools.bench.rtx4090_cache_audit", "--mode", mode,
                  "--output", str(destination), "--repeats", "5", "--queue-stress"]
        status["client_command"] = client
        with (a.out / f"client-{mode}.log").open("w", encoding="utf-8") as client_log:
            result = subprocess.run(client, stdout=client_log, stderr=subprocess.STDOUT)
        status["client_returncode"] = result.returncode
        settled_deadline = time.monotonic() + 5
        while True:
            with urllib.request.urlopen("http://127.0.0.1:18989/stats", timeout=5) as response:
                status["final_stats"] = json.load(response)
            requests = status["final_stats"]["requests"]
            if all(requests[name] == 0 for name in
                   ("running", "waiting", "prefilling", "decode_ready", "materializing")):
                break
            if time.monotonic() >= settled_deadline:
                raise RuntimeError("requests did not settle after completion and cancellation")
            time.sleep(0.1)
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
    finally:
        status["forced_termination"] = False
        if process.poll() is None:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(0.2)
                if process.poll() is None:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                process.wait(timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                status["forced_termination"] = True
                process.terminate()
                process.wait(timeout=15)
        status["server_returncode"] = process.returncode
        log.close()
        (a.out / "cache-campaign.json").write_text(json.dumps(statuses, indent=2) + "\n",
                                                  encoding="utf-8")
    print(mode, status.get("client_returncode"), status.get("error"), flush=True)
raise SystemExit(1 if any(s.get("error") or s.get("client_returncode") != 0 for s in statuses) else 0)
