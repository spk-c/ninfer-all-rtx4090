import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = Path('C:/coding/ninfer-q4k-validation')
BUILD = ROOT / 'build-validation'
BUNDLE = BUILD / 'tests/ninfer_tests.exe'
SAN = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/compute-sanitizer/compute-sanitizer.exe')
MODEL = 'C:/ninfer-all/models/Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer'
results = json.loads((OUT / 'checks.json').read_text(encoding='utf-8')) if (OUT / 'checks.json').exists() else []

def run(name, command, env=None, timeout=1800):
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        return
    attempt = 1 + sum(r['name'] == name for r in results)
    start = time.time()
    settings = os.environ.copy()
    settings.update(env or {})
    print('START', name, flush=True)
    stem = name if attempt == 1 else name + f'.attempt{attempt}'
    with (OUT / (stem + '.log')).open('w', encoding='utf-8') as log:
        try:
            status = subprocess.run(list(map(str, command)), cwd=ROOT, env=settings,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            status = 'timeout'
    results.append(dict(name=name, attempt=attempt, log=stem+'.log', command=list(map(str, command)), environment=env,
                        exit_code=status, seconds=time.time()-start))
    (OUT / 'checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('DONE', name, status, round(time.time()-start, 2), flush=True)

run('mmq-store', [BUNDLE, 'ninfer_linear_gguf_mmq_store_test'])
run('dense-oracle', [BUNDLE, 'ninfer_linear_gguf_test'])
run('expert-delivery', [BUILD / 'tests/ninfer_expert_delivery_test.exe'])
for tool in ('memcheck', 'racecheck', 'synccheck'):
    run('sanitizer-' + tool, [SAN, '--tool', tool, '--error-exitcode', '99',
                            BUNDLE, 'ninfer_linear_gguf_mmq_store_test'])
run('sanitizer-racecheck-smoke', [SAN, '--tool', 'racecheck', '--error-exitcode', '99',
                                 BUNDLE, 'ninfer_linear_gguf_mmq_store_test'],
    {'NINFER_Q4K_RACECHECK_SMOKE': '1'})
run('sanitizer-expert-delivery', [SAN, '--tool', 'memcheck', '--error-exitcode', '99',
                                 BUILD / 'tests/ninfer_expert_delivery_test.exe'])
run('sanitizer-expert-alignment', [SAN, '--tool', 'memcheck', '--error-exitcode', '99',
                                  BUILD / 'tests/ninfer_expert_delivery_test.exe', '--alignment-only'])
run('dflash2-real', [BUNDLE, 'ninfer_qwen3_5_dflash2_real_test', '5', '1', '1', '1', 'int8', '1', '1'],
    {'NINFER_TEST_ARTIFACT': MODEL})
