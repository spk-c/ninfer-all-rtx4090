"""Replay fixed recorded prompts and compare the earliest public token/logprob divergence.

Raw prompts and responses remain local. No performance claim: requesting logprobs
can change round planning. Server lifetime is owned by this process alone.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def post(base, payload):
    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(payload).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def serve(command, folder, action, env=None):
    folder.mkdir(parents=True, exist_ok=False)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    with (folder / 'server.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                                startupinfo=startup)
        status = {'command': command, 'pid': proc.pid}
        try:
            for _ in range(720):
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited {proc.returncode}')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18989/health', timeout=1):
                        break
                except OSError:
                    time.sleep(.25)
            else:
                raise TimeoutError('server startup')
            return action()
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    status['forced_termination'] = True
                    proc.terminate()
                    proc.wait(timeout=10)
            status['returncode'] = proc.returncode
            (folder / 'server-status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--recording', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=17)
    p.add_argument('--spec', choices=['none', 'dflash2'], default='dflash2')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--no-logprobs', action='store_true')
    p.add_argument('--no-graph', action='store_true')
    p.add_argument('--splits', help='diagnostic build only: force the same comma-separated prefill frontiers')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.recording.read_text(encoding='utf-8'))['rows'][:args.count]
    results = {}
    base = 'http://127.0.0.1:18989'
    for mode in ['hybrid', 'disabled']:
        folder = args.out / mode
        command = [str(args.server.resolve()), str(args.model.resolve()), '--host', '127.0.0.1',
                   '--port', '18989', '--model-id', 'audit', '--max-context', '32768',
                   '--kv-capacity', '32768', '--kv-dtype', 'rk4v4', '--gdn-state-fp16',
                   '--prefill-chunk', '1024', '--spec', args.spec,
                   '--max-concurrency', '2', '--no-thinking',
                   '--log-stats-interval-ms', '0']
        if args.spec != 'none':
            command += ['--draft-tokens', '5', '--lm-head-draft']
        command += ['--host-cache-mib', '256', '--use-alt-prefix-caching'] if mode == 'hybrid' else ['--no-prefix-reuse']
        if args.no_graph:
            command += ['--no-cuda-graph']
        env = os.environ.copy()
        if args.trace:
            env['NINFER_TARGET_TRACE'] = str((folder / 'trace').resolve())
        if args.splits:
            env['NINFER_DIAGNOSTIC_SPLITS'] = args.splits
        def replay():
            out = []
            for i, row in enumerate(rows):
                body = copy.deepcopy(row['request'])
                body['stream'] = False
                body.pop('stream_options', None)
                if not args.no_logprobs:
                    body.update(logprobs=True, top_logprobs=20)
                result = post(base, body)
                out.append({'label': row['label'], 'request': body, 'response': result})
                (folder / 'responses.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
                print(mode, i, row['label'], result.get('usage'), flush=True)
            return out
        results[mode] = serve(command, folder, replay, env)
    comparisons = []
    for a, b in zip(results['hybrid'], results['disabled']):
        x, y = a['response']['choices'][0], b['response']['choices'][0]
        ax, by = (x.get('logprobs') or {}).get('content', []), (y.get('logprobs') or {}).get('content', [])
        first = next((i for i, (s, t) in enumerate(zip(ax, by)) if s.get('token_id', s['token']) != t.get('token_id', t['token'])),
                     min(len(ax), len(by)) if len(ax) != len(by) else None)
        item = {'label': a['label'], 'text_equal': x['message'] == y['message'],
                'first_token_difference_zero_based': first, 'logprob_tokens': [len(ax), len(by)],
                'hybrid_usage': a['response'].get('usage'), 'disabled_usage': b['response'].get('usage')}
        if first is not None:
            item.update(hybrid_first=ax[first] if first < len(ax) else None,
                        disabled_first=by[first] if first < len(by) else None)
        comparisons.append(item)
    (args.out / 'summary.json').write_text(json.dumps({'comparisons': comparisons}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
