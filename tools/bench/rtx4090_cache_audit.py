"""Exercise a local OpenAI server with reproducible coding prompts and cache telemetry.

Run against separately started, otherwise matched servers with and without
--no-prefix-reuse. TTFT is measured only for SSE content/reasoning deltas;
aggregate response latency is explicitly not TTFT. Raw responses stay local.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request


def request(base: str, payload: dict, *, cancel: bool = False,
            cancel_prefill: bool = False) -> dict:
    start = time.perf_counter()
    first = None
    text = ""
    usage, timings = {}, {}
    chunks = []
    done = False
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        if payload.get("stream"):
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                chunks.append(chunk)
                usage = chunk.get("usage") or usage
                timings = chunk.get("timings") or timings
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    fragment = delta.get("content") or delta.get("reasoning_content") or ""
                    if fragment and first is None:
                        first = time.perf_counter() - start
                    text += fragment
                if cancel and first is not None:
                    break
                if cancel_prefill:
                    time.sleep(0.05)
                    break
        else:
            chunk = json.load(response)
            chunks.append(chunk)
            usage, timings = chunk.get("usage", {}), chunk.get("timings", {})
            message = chunk["choices"][0]["message"]
            text = (message.get("reasoning_content") or "") + (message.get("content") or "")
    if payload.get("stream") and not (cancel or cancel_prefill) and not done:
        raise RuntimeError("SSE connection ended without [DONE]")
    return {"stream": bool(payload.get("stream")), "cancelled_by_client": cancel or cancel_prefill,
            "cancelled_before_first_content": cancel_prefill and first is None,
            "ttft_s": first, "response_s": time.perf_counter() - start,
            "text": text, "usage": usage, "timings": timings, "responses": chunks}


def queue_stress(base: str) -> dict:
    """Requires two lanes, one pending slot and a finite pending timeout on the server."""
    release = threading.Event()
    ready = [threading.Event(), threading.Event()]
    body = {"model": "audit", "messages": [{"role": "user", "content":
        "Print all integers from 1 to 1000000, one per line. Start immediately and continue."}],
        "temperature": 0, "max_tokens": 4096, "ignore_eos": True, "stream": True}

    def blocker(index):
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            for line in response:
                if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                    chunk = json.loads(line[6:])
                    if "error" in chunk:
                        raise RuntimeError(chunk["error"])
                    if any(c.get("delta", {}).get("content") for c in chunk.get("choices", [])):
                        ready[index].set()
                        release.wait(20)
                        return
        raise RuntimeError("blocker completed before its first content")

    def probe(stream):
        started = time.perf_counter()
        payload = {**body, "max_tokens": 1, "stream": stream}
        try:
            result = request(base, payload)
            return {"unexpected_success": True, "result": result}
        except urllib.error.HTTPError as error:
            return {"http_status": error.code, "body": error.read().decode(),
                    "wall_s": time.perf_counter() - started}
        except RuntimeError as error:
            return {"sse_error": str(error), "wall_s": time.perf_counter() - started}

    with ThreadPoolExecutor(max_workers=3) as pool:
        blockers = [pool.submit(blocker, i) for i in range(2)]
        try:
            if not all(event.wait(15) for event in ready):
                raise RuntimeError("could not occupy both active lanes")
            queued = pool.submit(probe, True)
            time.sleep(0.2)
            overloaded = probe(False)
            timeout = queued.result(timeout=15)
        finally:
            release.set()
            for future in blockers:
                future.result(timeout=15)
    return {"overload": overloaded, "queued_timeout": timeout}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18989")
    parser.add_argument("--mode", choices=("hybrid", "disabled"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--queue-stress", action="store_true",
                        help="server must have two lanes, one pending slot and <=10 s pending timeout")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a new path to preserve measurements")
    rows, failures = [], []
    queue_report = None

    def run(label: str, messages: list[dict], key: str | None, stream=False, cancel=False,
            cancel_prefill=False):
        payload = {"model": "audit", "messages": messages, "max_tokens": 32,
                   "temperature": 0, "stream": stream, "prompt_cache_key": key,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if key is None:
            payload.pop("prompt_cache_key")
        if stream:
            payload["stream_options"] = {"include_usage": True}
        def counters():
            with urllib.request.urlopen(args.url + "/stats", timeout=5) as response:
                return json.load(response)["counters"]

        attributable = not label.startswith("concurrent/")
        before = counters() if attributable else None
        result = request(args.url, payload, cancel=cancel, cancel_prefill=cancel_prefill)
        if cancel or cancel_prefill:
            settle_started = time.perf_counter()
            while True:
                with urllib.request.urlopen(args.url + "/stats", timeout=5) as response:
                    state = json.load(response)["requests"]
                if all(state[name] == 0 for name in ("running", "waiting", "materializing")):
                    break
                if time.perf_counter() - settle_started > 30:
                    raise TimeoutError("cancelled request did not release its Engine resources")
                time.sleep(0.05)
            result["cancellation_settle_s"] = time.perf_counter() - settle_started
        after = counters() if attributable else None
        result["engine_counter_deltas"] = ({name: after[name] - before[name] for name in
            ("computed_prefill_tokens", "reused_prompt_tokens")} if attributable else None)
        result["engine_counter_scope"] = ("serial request, including cancellation settlement" if attributable
                                         else "concurrent; not attributable")
        memory = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        result["device_wide_memory_used_mib"] = memory.stdout.strip()
        row = result
        # Histories grow in place between turns. Preserve the request that was actually sent,
        # rather than letting later appends rewrite earlier measurement records.
        row.update({"label": label, "key": key, "request": json.loads(json.dumps(payload))})
        rows.append(row)
        print(label, "TTFT", result["ttft_s"], "timings", result["timings"], flush=True)
        return result

    def check(condition, message):
        if not condition:
            failures.append(message)

    try:
        for stream in (False, True):
            for rep in range(args.repeats):
                # Vary the initial system block to make each pair genuinely cold.
                system = (f"You are a coding assistant. Audit case {stream}/{rep}.\n" +
                          "Preserve public interfaces. Explain concrete bugs and their fixes.\n" * 40)
                code = "\n".join(f"def function_{i}(x): return x + {i}" for i in range(200))
                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": code + "\nSummarize in one sentence."}]
                key = f"audit-{stream}-{rep}"
                cold = run(f"{stream}/{rep}/cold", messages, key, stream)
                warm = run(f"{stream}/{rep}/warm", messages, key, stream)
                check(cold["usage"].get("prompt_tokens") == warm["usage"].get("prompt_tokens"),
                      f"{key}: identical prompts have different token counts")
                reused = warm["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                check(reused > 0 if args.mode == "hybrid" else reused == 0,
                      f"{key}: unexpected cached token count {reused}")
                check(bool(warm["timings"]), f"{key}: final timing telemetry is missing")
                # Record exact output equality as a finding; quantized recurrent-state
                # prefix boundaries can change greedy tokens, so do not hide differences.
                warm["matches_cold_text"] = warm["text"] == cold["text"]
                tools = messages + [
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "audit_call", "type": "function", "function": {
                            "name": "read_file", "arguments": '{"path":"example.py"}'}}]},
                    {"role": "tool", "tool_call_id": "audit_call", "content": "def add(a,b): return a+b"}]
                if rep == 0:
                    run(f"{stream}/tool-turn", tools, key, stream)
                    run(f"{stream}/tool-repeat", tools, key, stream)
                    run(f"{stream}/tool-no-key", tools, None, stream)
                    tool_chain = tools + [
                        {"role": "assistant", "content": None, "tool_calls": [
                            {"id": "audit_call_2", "type": "function", "function": {
                                "name": "read_file", "arguments": '{"path":"test_example.py"}'}}]},
                        {"role": "tool", "tool_call_id": "audit_call_2",
                         "content": "def test_add(): assert add(1,2) == 3"}]
                    run(f"{stream}/second-tool-turn", tool_chain, key, stream)
                    run(f"{stream}/second-tool-repeat", tool_chain, key, stream)
                    changed = [{"role": "system", "content": "Reply with the word ISOLATED only."},
                               {"role": "user", "content": "Ignore all previous code. Reply ISOLATED."}]
                    run(f"{stream}/invalidated", changed, key, stream)
                    run(f"{stream}/other-session", changed, "other-session", stream)
        concurrent_messages = [{"role": "system", "content": "You are a concise coding assistant."},
                               {"role": "user", "content": "Explain binary search in one sentence."}]
        session_messages = [[{"role": "system", "content": f"Reply with {word} only."},
                             {"role": "user", "content": f"Return exactly {word}."}]
                            for word in ("ALPHA", "BETA")]
        references = [run(f"concurrent-reference/{i}", messages, f"concurrent-{i}")
                      for i, messages in enumerate(session_messages)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            calls = [pool.submit(run, f"concurrent/{i}", session_messages[i],
                                 f"concurrent-{i}", True) for i in range(2)]
            for i, call in enumerate(calls):
                actual = call.result()
                check(actual["text"] == references[i]["text"],
                      f"concurrent session {i} differs from its own fresh reference")
        long_history = [{"role": "system", "content": "You are a concise coding assistant."},
                        {"role": "user", "content": "\n".join(
                            f"def long_function_{i}(x): return x * {i}" for i in range(1200)) +
                            "\nSummarize this module in one sentence."}]
        for turn in range(4):
            result = run(f"long-session/{turn}", long_history, "long-session", True)
            long_history += [{"role": "assistant", "content": result["text"]},
                             {"role": "user", "content": f"Review function {turn} for bugs briefly."}]
        run("long-session/retry", long_history, "long-session", True)
        run("cancel-stream", concurrent_messages, "cancel", True, True)
        run("after-cancel", concurrent_messages, "cancel", True)
        cancelled_history = [{"role": "system", "content": "Cancellation audit: explain the Python module."},
                             long_history[1]]
        run("cancel-prefill", cancelled_history, "cancel-prefill", True, cancel_prefill=True)
        time.sleep(0.5)
        run("after-prefill-cancel", cancelled_history, "cancel-prefill", True)
        if args.queue_stress:
            queue_report = queue_stress(args.url)
            check(queue_report["overload"].get("http_status") == 429,
                  "full admission queue did not return HTTP 429")
            check("request_queue_timeout" in json.dumps(queue_report["queued_timeout"]),
                  "queued SSE request did not expose its admission timeout")
            time.sleep(0.5)
            run("after-queue-stress", concurrent_messages, "after-queue", True)
        invalid = {"model": "audit", "messages": concurrent_messages, "prompt_cache_key": 42}
        try:
            request(args.url, invalid)
            failures.append("numeric prompt_cache_key was accepted")
        except urllib.error.HTTPError as error:
            check(error.code == 400, f"numeric key returned HTTP {error.code}")
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"mode": args.mode, "repeats": args.repeats,
            "ttft_definition": "First nonempty SSE content or reasoning delta; aggregate TTFT unavailable",
            "rows": rows, "queue_stress": queue_report, "failures": failures}, indent=2) + "\n", encoding="utf-8")
    print("failures:", failures, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
