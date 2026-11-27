import io
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from tools.bench import serve_load


CONTENT = b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
FINISH = b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'
USAGE = b'data: {"usage":{"prompt_tokens":10,"completion_tokens":2}}\n\n'
DONE = b'data: [DONE]\n\n'


@pytest.mark.parametrize(("events", "error"), [
    (CONTENT + FINISH + USAGE + DONE, None),
    (CONTENT + FINISH + USAGE, "before [DONE]"),
    (CONTENT + FINISH + DONE, "without completion token usage"),
    (CONTENT + USAGE + DONE, "without a finish reason"),
])
def test_stream_terminal_events(monkeypatch, events, error):
    response = io.BytesIO(events)
    response.status = 200
    connection = SimpleNamespace(
        request=lambda *args, **kwargs: None,
        getresponse=lambda: response,
        close=lambda: None,
    )
    monkeypatch.setattr(serve_load.http.client, "HTTPConnection", lambda *a, **kw: connection)
    row = serve_load.stream_chat(urlparse("http://127.0.0.1"), {}, timeout=1)
    args = SimpleNamespace(max_tokens=2, natural_eos=True)
    summary = serve_load.summarize("chat", 1, [row], row["sent"], row["ended"], args, None)
    if error is None:
        assert row["error"] is None
        assert summary["completed"] == 1
        assert summary["generated_tokens"] == 2
        assert summary["latency_ms"]["p95"] is not None
    else:
        assert error in row["error"]
        assert summary["completed"] == 0
        assert summary["error_count"] == 1
        assert summary["generated_tokens"] == 0
        assert summary["latency_ms"]["p95"] is None
