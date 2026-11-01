"""Run installed Qwen Code in an isolated fixture, with actual session resumes/tools."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid
from tools.bench.rtx4090_cache_differential import serve


SOURCE = '''def average(values):
    return sum(values) / len(values)

def interval(start, stop):
    return list(range(start, stop))

def bucket(value, width):
    return int(value / width)

def unique(values):
    return list(set(values))

def parse_record(text):
    return text.split(',')
'''
TESTS = '''import unittest
import ledger
class LedgerTest(unittest.TestCase):
    def test_average(self):
        self.assertIsNone(ledger.average([]))
        self.assertEqual(ledger.average([1, 2, 3]), 2)
    def test_interval(self):
        self.assertEqual(ledger.interval(2, 4), [2, 3, 4])
        self.assertEqual(ledger.interval(3, 3), [3])
        self.assertEqual(ledger.interval(4, 2), [])
    def test_bucket(self):
        for x in range(-100, 101):
            self.assertEqual(ledger.bucket(x, 7), x // 7)
    def test_unique(self):
        self.assertEqual(ledger.unique(['z', 'a', 'z', 'b', 'a']), ['z', 'a', 'b'])
        self.assertEqual(ledger.unique([[1], [2], [1]]), [[1], [2]])
    def test_csv(self):
        self.assertEqual(ledger.parse_record('"a,b",c,"d""e"'), ['a,b', 'c', 'd"e'])
if __name__ == '__main__': unittest.main()
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--client', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['hybrid', 'disabled'], default='hybrid')
    a = p.parse_args()
    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    workspace = root / 'workspace'
    workspace.mkdir()
    (workspace / 'ledger.py').write_text(SOURCE)
    (workspace / 'test_ledger.py').write_text(TESTS)
    spec = ('Ledger contracts: empty average returns None; interval includes both ends; '
            'bucket is mathematical floor; unique preserves order and accepts unhashable values; '
            'parse_record uses CSV quoting. Do not edit tests or this specification.\n')
    spec += '\n'.join(f'Historical event {i:04}: account A{i%11}, delta {i*17%101-50}, month {i%12+1:02}; retain source order.' for i in range(140))
    (workspace / 'SPEC.md').write_text(spec)
    env = os.environ.copy()
    env['QWEN_HOME'] = str(root / 'qwen-home')
    env['QWEN_RUNTIME_DIR'] = str(root / 'qwen-runtime')
    env['OPENAI_API_KEY'] = 'local-audit'
    env['OPENAI_BASE_URL'] = 'http://127.0.0.1:18989/v1'
    env['QWEN_CODE_MAX_OUTPUT_TOKENS'] = '2048'
    home = Path(env['QWEN_HOME']); home.mkdir()
    settings = {'telemetry': {'enabled': False}, 'model': {'generationConfig': {
        'contextWindowSize': 32768, 'samplingParams': {'temperature': 0, 'max_tokens': 2048}}},
        'tools': {'disabled': ['agent', 'send_message', 'list_agents', 'web_fetch', 'web_search']}}
    (home / 'settings.json').write_text(json.dumps(settings))
    sid = str(uuid.uuid4())
    commands = []
    prompts = [
        'Work only inside this fixture. Do not delegate, access network, or use git. '
        f'Use "{a.python}" for Python. Read SPEC.md, ledger.py and test_ledger.py using tools. '
        'Run the tests to observe failures. Fix only average, then rerun test_average. '
        'Do not change tests. Report the observed remaining failures.',
        'Continue the same coding session. Re-read the relevant function and tests. '
        'Fix interval and bucket according to SPEC.md. Run their two tests and report actual results. '
        'Leave unique and CSV for the next turn. Do not change tests.',
        'Continue. Read the remaining unique and CSV contracts/tests. Implement them, preserving '
        'first occurrence and supporting unhashable values. Run the full unittest suite and inspect '
        'the final source with tools. Do not change tests or SPEC.md.',
        'Final verification in this resumed session: read ledger.py again, run the full tests, '
        'and use a separate Python command to check bucket(x,7)==x//7 for every integer -10000 '
        'through 10000. Read the last five historical events in SPEC.md and report their account '
        'IDs. Base the answer on actual tool outputs; do not modify more files.'
    ]
    server = [str(a.server.resolve()), str(a.model.resolve()), '--host', '127.0.0.1', '--port', '18989',
              '--model-id', 'audit', '--max-context', '32768', '--kv-capacity', '32768',
              '--kv-dtype', 'rk4v4', '--gdn-state-fp16', '--prefill-chunk', '1024',
              '--spec', 'dflash2', '--draft-tokens', '5', '--lm-head-draft', '--max-concurrency', '2',
              '--no-thinking', '--log-stats-interval-ms', '0', '--request-log-jsonl', str(root / 'requests.jsonl')]
    server += ['--use-alt-prefix-caching', '--host-cache-mib', '256'] if a.mode == 'hybrid' else ['--no-prefix-reuse']
    def run():
        for i, prompt in enumerate(prompts):
            cmd = ['node', str(a.client.resolve()), '--bare', '--auth-type', 'openai',
                   '--openai-base-url', env['OPENAI_BASE_URL'], '--openai-api-key', 'local-audit',
                   '--model', 'audit', '--approval-mode', 'yolo', '--output-format', 'stream-json',
                   '--max-session-turns', '30', '--max-tool-calls', '60', '--max-wall-time', '10m',
                   '--openai-logging', '--openai-logging-dir', str(root / 'http'),
                   '--session-id' if i == 0 else '--resume', sid, '--prompt', prompt]
            with (root / f'turn-{i}.jsonl').open('w', encoding='utf-8') as output, (root / f'turn-{i}.stderr').open('w', encoding='utf-8') as error:
                result = subprocess.run(cmd, cwd=workspace, env=env, stdout=output, stderr=error, timeout=660)
            commands.append({'turn': i, 'command': cmd, 'exit_code': result.returncode})
            (root / 'client-status.json').write_text(json.dumps(commands, indent=2))
            print(a.mode, 'turn', i, 'exit', result.returncode, flush=True)
            if result.returncode:
                raise RuntimeError(f'Qwen turn {i} failed; inspect preserved stderr/JSON')
        result = subprocess.run([str(a.python), '-m', 'unittest', '-v'], cwd=workspace, capture_output=True, text=True)
        (root / 'independent-tests.txt').write_text(result.stdout + result.stderr)
        if result.returncode or (workspace / 'test_ledger.py').read_text() != TESTS or (workspace / 'SPEC.md').read_text() != spec:
            raise RuntimeError('independent tests or fixture immutability failed')
    serve(server, root / 'server', run)


if __name__ == '__main__':
    main()
