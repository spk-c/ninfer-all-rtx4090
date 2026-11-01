"""Compare valid logical cache bytes and aligned raw target logits from diagnostic builds."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.bench.analyze_qwen_target_trace import groups, tensor, compare, floats, audit_verification


def records(folder, name='index.jsonl'):
    return [json.loads(line) for line in (folder / name).read_text().splitlines()]


def compare_state(folder_a, kind_a, folder_b, kind_b):
    def collect(folder, kind):
        out = {}
        for r in records(folder):
            if r['kind'].startswith(kind + '-'):
                out.setdefault((r['kind'][len(kind):], r['name']), r)
        for r in records(folder, 'kv.jsonl'):
            if r['kind'] == kind:
                out.setdefault(('kv-' + str(r['layer']), r['name']), r)
        return out
    a, b = collect(folder_a, kind_a), collect(folder_b, kind_b)
    if not a or a.keys() != b.keys():
        raise ValueError('state inventories missing or inconsistent')
    rows = []
    for key in a:
        x, y = (folder_a/a[key]['file']).read_bytes(), (folder_b/b[key]['file']).read_bytes()
        if len(x) != len(y):
            raise ValueError('state extent mismatch')
        row = {'component': list(key), 'bytes': len(x), 'exact': x == y}
        if x != y:
            ax, by = np.frombuffer(x, dtype=np.uint8), np.frombuffer(y, dtype=np.uint8)
            row['different_bytes'] = int(np.count_nonzero(ax != by))
            row['first_different_byte'] = int(np.flatnonzero(ax != by)[0])
            dtype = a[key]['dtype']
            if dtype in [0, 1, 6] and not key[0].startswith('kv-'):
                def decode(data):
                    if dtype == 0:
                        return (np.frombuffer(data, np.uint16).astype(np.uint32) << 16).view(np.float32)
                    return np.frombuffer(data, np.float32 if dtype == 1 else np.float16).astype(np.float32)
                av, bv = decode(x), decode(y)
                row['nonfinite_elements'] = int(np.count_nonzero(~np.isfinite(av) | ~np.isfinite(bv)))
                delta = np.abs(av-bv)
                row['max_abs_delta'] = float(np.max(delta))
                row['rms_delta'] = float(np.sqrt(np.mean(delta.astype(np.float64)**2)))
        rows.append(row)
    return {'from': str(folder_a), 'from_kind': kind_a, 'to': str(folder_b), 'to_kind': kind_b,
            'all_exact': all(r['exact'] for r in rows), 'bytes': sum(r['bytes'] for r in rows), 'components': rows}


def requests(folder):
    out = []
    current = None
    for group in groups(folder).values():
        kind = group['kind']
        if kind == 'prefill':
            logits = tensor(folder, group['logits'])
            values = floats(logits)
            first = max(range(len(values)), key=lambda k: values[k])
            current = {'tokens': [first], 'logits': [logits], 'audits': [], 'prompt_tokens': None}
            out.append(current)
        elif current is not None and kind == 'plain':
            if current['prompt_tokens'] is None:
                current['prompt_tokens'] = tensor(folder, group['positions'])[0]
            current['tokens'].append(tensor(folder, group['sampled'])[0])
            current['logits'].append(tensor(folder, group['logits']))
        elif current is not None and kind == 'verify':
            if current['prompt_tokens'] is None:
                current['prompt_tokens'] = tensor(folder, group['positions'])[0]
            width = tensor(folder, group['valid'])[0]
            count = tensor(folder, group['counts'])[0]
            logits = tensor(folder, group['logits'])
            vocab = group['logits']['shape'][0]
            audit = audit_verification(tensor(folder, group['positions']), logits, vocab,
                tensor(folder, group['targets']), tensor(folder, group['drafts']),
                tensor(folder, group['licensed']), count, tensor(folder, group['accepted'])[0], width)
            current['audits'].append(audit)
            current['tokens'].extend(tensor(folder, group['licensed'])[:count])
            current['logits'].extend(logits[j*vocab:(j+1)*vocab] for j in range(count))
    return out


def compare_requests(a, b, count=32):
    out = []
    if len(a) != len(b):
        raise ValueError('request count differs')
    for x, y in zip(a, b):
        xt, yt = x['tokens'][:count], y['tokens'][:count]
        first = next((i for i,(s,t) in enumerate(zip(xt,yt)) if s != t),
                     min(len(xt),len(yt)) if len(xt) != len(yt) else None)
        limit = min(count, len(x['logits']), len(y['logits']), first+1 if first is not None else count)
        differences = [{'generated_token_one_based': i+1, **compare(x['logits'][i], y['logits'][i])}
                       for i in range(limit)]
        out.append({'tokens_equal': xt==yt, 'first_token_difference_one_based': None if first is None else first+1,
                    'a_tokens': xt, 'b_tokens': yt,
                    'first_logit_difference': next((r for r in differences if r['different_elements']), None),
                    'at_first_token_difference': differences[first] if first is not None and first < len(differences) else None,
                    'same_prefix_logit_comparisons': differences,
                    'verification_errors_a': [r for r in x['audits'] if r['errors']],
                    'verification_errors_b': [r for r in y['audits'] if r['errors']]})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hybrid', type=Path, required=True)
    p.add_argument('--disabled', type=Path, required=True)
    p.add_argument('--frontier', type=int, default=3529)
    p.add_argument('--prompt-tokens', type=int, required=True,
                   help='explicit HTTP workload extent; excludes startup calibration forwards')
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    if a.out.exists(): p.error('output exists')
    h, d = requests(a.hybrid), requests(a.disabled)
    result={'requests':compare_requests([r for r in h if r['prompt_tokens'] == a.prompt_tokens],
                                       [r for r in d if r['prompt_tokens'] == a.prompt_tokens]),
            'excluded_startup_requests': [sum(r['prompt_tokens'] != a.prompt_tokens for r in arm) for arm in [h,d]],
            'snapshot_to_resume':compare_state(a.hybrid,f'chunk-{a.frontier}',a.hybrid,f'resume-{a.frontier}'),
            'prefill_frontiers_hybrid': sorted({int(r['kind'].split('-')[1]) for r in records(a.hybrid) if r['kind'].startswith('chunk-')}),
            'prefill_frontiers_disabled': sorted({int(r['kind'].split('-')[1]) for r in records(a.disabled) if r['kind'].startswith('chunk-')})}
    a.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
