import json
from pathlib import Path
import sys

ROOT = Path('C:/coding/ninfer-q4k-validation')
sys.path.insert(0, str(ROOT / 'tools/bench'))
from compare_gguf_iq3s_quality_4090 import PROMPTS, run, write_backend_comparison

OUT = Path(__file__).resolve().parent / 'quality'
OUT.mkdir(exist_ok=True)
MODEL = Path('C:/ninfer-all/models/Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer')
arm = sys.argv[1]
exe = (Path('C:/coding/ninfer-all/build-rtx4090/apps/ninfer.exe') if arm == 'base'
       else ROOT / 'build-validation/apps/ninfer.exe')
saved = OUT / (arm + '.json')
results = {}
for name, prompt in PROMPTS.items():
    prompt_file = OUT / (name + '.messages.json')
    prompt_file.write_text(json.dumps([{'role': 'user', 'content': prompt}]), encoding='utf-8')
    for mode in ('none', 'dflash2'):
        results[name + '-' + mode] = run(exe, MODEL, prompt_file, OUT, name, mode, arm, True, 1024)
        saved.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(arm, name, mode, 'completed', flush=True)
if (OUT / 'base.json').exists() and (OUT / 'candidate.json').exists():
    base = json.loads((OUT / 'base.json').read_text(encoding='utf-8'))
    candidate = json.loads((OUT / 'candidate.json').read_text(encoding='utf-8'))
    summary = {}
    for name in base:
        a, b = base[name], candidate[name]
        equal = all(a[key] == b[key] for key in ('generated_ids', 'text', 'prompt_tokens'))
        summary[name] = dict(equal=equal, base=a, candidate=b)
        print(name, 'equal' if equal else 'DIFFERENT', flush=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_backend_comparison(summary, OUT)
    if not all(v['equal'] for v in summary.values()):
        sys.exit(1)
