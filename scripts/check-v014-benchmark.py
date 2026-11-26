"""Validate published evidence and compare a new RTX4090 long-context campaign."""
from pathlib import Path
import argparse, csv, json, math, statistics, hashlib, sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'profiles/bench/rtx4090-v014-long-coding-20261126'
FIRST=ROOT/'profiles/bench/rtx4090-v014-tuning-20261126'
CONTEXTS=(65536,131072,176128)
KS=(5,7,9,11,15)
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def need(ok,message):
    if not ok:raise ValueError(message)
def engine(path,n):
    rows=load(path/'engine-results.json')
    need(len(rows)==n,'wrong Engine count')
    keys=set()
    for r in rows:
        key=(r['group'],r['tag'],r['ctx'],r['rep'])
        need(key not in keys,'duplicate measurement');keys.add(key)
        for k in ('prefill','decode','total_seconds'):
            need(math.isfinite(r[k]) and r[k]>0,'invalid '+k)
        need(0<=r['accept']<=1,'invalid acceptance')
        report=path/'engine'/r['report']
        if not report.exists():report=path/r['report']
        d=load(report);t=d['tests'][0]
        need(t['reps'][0]['decode_output_tokens']==512,'incomplete output')
        need(d['environment']['gpu_name']=='NVIDIA GeForce RTX 4090','different GPU')
        need(d['config']['kv_cache']=='rk4v4','different KV')
        need(d['config']['draft_tokens']==r['k'] and d['config']['prefill_chunk']==r['chunk'],'configuration mismatch')
        need(t['n_prompt']==r['ctx']-1024,'wrong prompt length')
        need(math.isclose(t['decode_output_tok_s_mean'],r['decode'],rel_tol=1e-8),'decode mismatch')
    return rows
def long_cells(path):
    rows=engine(path,30)
    cells={}
    for ctx in CONTEXTS:
        for k in KS:
            a=[r for r in rows if r['ctx']==ctx and r['k']==k]
            need(len(a)==2 and {r['rep'] for r in a}=={1,2},'missing cell')
            need(all(r['chunk']==4096 and not r['adaptive'] and not r['cublas'] and r['graph']=='default' for r in a),'incomparable long cell')
            cells[ctx,k]=statistics.median(r['decode'] for r in a)
    return cells
def validate(path1,path2):
    a=load(path1/'manifest.json');b=load(path2/'manifest.json')
    for k in ('server_sha256','bench_sha256','runtime_sha256','model_sha256','corpus_sha256'):
        need(a[k]==b[k] and len(a[k])==64,'identity mismatch '+k)
    seed=(ROOT/'bench/fixtures/bench_corpus.ids').read_text().split()
    corpus=(' '.join((seed*3)[:176128])+'\r\n').encode('ascii')
    need(hashlib.sha256(corpus).hexdigest()==a['corpus_sha256'],'fixture mismatch')
    engine(path1,48);cells=long_cells(path2)
    ranking=load(path2/'draft-ranking.json')
    for r in ranking['results']:
        need(math.isclose(cells[r['ctx'],r['k']],r['decode'],rel_tol=1e-9),'ranking mismatch')
    need(all(max(KS,key=lambda k:cells[c,k])==7 for c in CONTEXTS),'K7 reference mismatch')
    for path,count,eng in ((path1,33,48),(path2,16,30)):
        summary=load(path/'summary.json')
        need(summary['status']=='finished' and not summary['errors'] and summary['engine_runs']==eng,'invalid summary')
        rows=load(path/'server-results.json')
        need(len(rows)==count,'wrong API count')
        for r in rows:need(r['wall_seconds']>0 and r['output_tokens']>0,'incomplete API')
    over=load(path2/'concurrency-results.json')
    need(len(over)==8 and all(x['offered_overlap_confirmed'] and len(x['requests'])==2 for x in over),'overlap mismatch')
    buck=load(path2/'buckets-results.json')
    need(len(buck)==4 and len({x['token_ids_sha256'] for x in buck})==1,'bucket token mismatch')
    print('Reference evidence verified: 78 Engine + 49 nonstream + 16 streaming = 143 measurements.')
    return cells
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,help='New complete long-context campaign directory')
    p.add_argument('--tolerance',type=float,default=.10,help='Allowed relative slowdown (default 10%%)')
    a=p.parse_args()
    need(0<=a.tolerance<1,'invalid tolerance')
    ref=validate(FIRST,BASE)
    if a.candidate:
        old=load(BASE/'manifest.json');new=load(a.candidate/'manifest.json')
        for k in ('model_sha256','corpus_sha256'):
            need(old[k]==new[k],'different '+k)
        summ=load(a.candidate/'summary.json')
        need(summ['status']=='finished' and not summ['errors'],'candidate not complete')
        cand=long_cells(a.candidate)
        failures=[]
        for ctx in CONTEXTS:
            change=cand[ctx,7]/ref[ctx,7]-1
            print(f'K7 {ctx}: {ref[ctx,7]:.2f} -> {cand[ctx,7]:.2f} tok/s ({change:+.2%})')
            if change<-a.tolerance:failures.append(str(ctx))
        print('Screening threshold only; investigate GPU telemetry, driver, output correctness and variance before attribution.')
        if failures:raise ValueError('Decode regression at '+', '.join(failures))
if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError) as exc:print(f'FAIL: {exc}',file=sys.stderr);sys.exit(1)
