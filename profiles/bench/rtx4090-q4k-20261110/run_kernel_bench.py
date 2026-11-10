import csv
import json
import os
from pathlib import Path
import statistics
import subprocess

OUT = Path(__file__).resolve().parent / 'kernel-bench'
OUT.mkdir(exist_ok=True)
EXE = Path('C:/coding/ninfer-q4k-validation/build-validation/bench/q4k_qualification_linear.exe')
records = []
for n, k in ((128, 5120), (17408, 5120), (5120, 17408)):
    for t in (64, 65, 128, 129, 512, 2048, 4096):
        for pair in (1, 2):
            for route in (('compiled', 'async') if pair == 1 else ('async', 'compiled')):
                stem = f'n{n}-k{k}-t{t}-p{pair}-{route}'
                report = OUT / (stem + '.csv')
                cmd = [str(EXE), '--qtype', 'q4_k', '--n', str(n), '--k', str(k),
                       '--t', str(t), '--warmup', '3', '--repeat', '20',
                       '--csv-out', str(report)]
                env = os.environ.copy()
                env['NINFER_Q4K_QUALIFICATION_ROUTE'] = route
                with (OUT / (stem + '.log')).open('w', encoding='utf-8') as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=180, check=True)
                row = next(csv.DictReader(report.open(encoding='utf-8')))
                records.append(dict(n=n, k=k, t=t, pair=pair, route=route, result=row, command=cmd))
        arms = {r: [float(v['result']['median_us']) for v in records
                    if (v['n'], v['k'], v['t'], v['route']) == (n, k, t, r)]
                for r in ('compiled', 'async')}
        print(n, k, t, arms, flush=True)
        (OUT / 'records.json').write_text(json.dumps(records, indent=2), encoding='utf-8')

summary = []
for n, k in ((128, 5120), (17408, 5120), (5120, 17408)):
    for t in (64, 65, 128, 129, 512, 2048, 4096):
        arms = {r: [float(v['result']['median_us']) for v in records
                    if (v['n'], v['k'], v['t'], v['route']) == (n, k, t, r)]
                for r in ('compiled', 'async')}
        changes = [(a/b-1)*100 for a, b in zip(arms['async'], arms['compiled'])]
        summary.append(dict(n=n, k=k, t=t, compiled_us=arms['compiled'], async_us=arms['async'],
                            latency_change_pct=changes, median_change_pct=statistics.median(changes)))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
