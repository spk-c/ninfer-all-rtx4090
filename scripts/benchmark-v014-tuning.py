from pathlib import Path
import concurrent.futures as cf, csv, difflib, hashlib, json, os, random, re, statistics, subprocess as sp, sys, threading, time, urllib.request, urllib.error, uuid

import argparse
ROOT=BIN=MODEL=SOURCE=CORPUS_SRC=CORPUS=OUT=STOP=None
PORT=8099
URL=''
ALLOW_OTHER_BUILD=False
def configure(bin_dir,model,repo,out,corpus,port=8099,allow_other_build=False):
    global ROOT,BIN,MODEL,SOURCE,CORPUS_SRC,CORPUS,OUT,STOP,PORT,URL,ALLOW_OTHER_BUILD
    ALLOW_OTHER_BUILD=allow_other_build
    BIN=Path(bin_dir).resolve(); MODEL=Path(model).resolve(); SOURCE=Path(repo).resolve()
    OUT=Path(out).resolve()
    if OUT.exists() and any(OUT.iterdir()): raise RuntimeError("Output directory must be empty")
    OUT.mkdir(parents=True,exist_ok=True)
    CORPUS_SRC=Path(corpus).resolve() if corpus else SOURCE/'bench/fixtures/bench_corpus.ids'
    CORPUS=OUT/'long-context-176k.ids'; STOP=OUT/'STOP'
    PORT=port; URL=f'http://127.0.0.1:{port}'
def cli():
    p=argparse.ArgumentParser(description="Reproduce RTX4090 v0.1.4 tuning campaign")
    p.add_argument('--bin-dir',required=True,type=Path);p.add_argument('--model',required=True,type=Path)
    p.add_argument('--repo',default=Path(__file__).resolve().parents[1],type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--corpus',type=Path)
    p.add_argument('--port',type=int,default=8099)
    p.add_argument('--allow-other-build',action='store_true',help='Explicitly allow a later release for regression screening')
    a=p.parse_args();configure(a.bin_dir,a.model,a.repo,a.out,a.corpus,a.port,a.allow_other_build)
FLAGS=['NINFER_SM89_ATTENTION_BUCKETS','NINFER_SM89_FLAT_DRAFT_GRAPHS','NINFER_SM89_FLAT_TARGET_GRAPHS','NINFER_SM89_STARTUP_CALIBRATION','NINFER_SM89_PHYSICAL_STARTUP_CALIBRATION','NINFER_SM89_LIVE_CONTEXT_REPLAY']
rows=[]
serv=[]
errors=[]
lock=threading.RLock()
def stamp(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')
def event(phase, **data):
    r={'time':stamp(),'phase':phase,**data}
    with (OUT/'progress.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps(r,ensure_ascii=False),flush=True)
def save(name,data):
    p=OUT/name
    t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(t,p)
def digest(p):
    with p.open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def guard():
    if STOP.exists(): raise InterruptedError('STOP requested')
    s=sp.check_output(['tasklist','/FO','CSV','/NH'],text=True,errors='replace')
    names={r[0].lower() for r in csv.reader(s.splitlines()) if r}
    if names.intersection({'ninfer-serve.exe','ninfer_bench.exe','ninfer-calibrate.exe','ninfer.exe'}):
        raise InterruptedError('Other NInfer process is active; refusing GPU contention')
    try:
        import socket
        with socket.create_connection(('127.0.0.1',PORT),timeout=.3): raise InterruptedError('Port 8099 occupied')
    except OSError: pass
def environment(graph='default'):
    e=os.environ.copy()
    for flag in FLAGS: e.pop(flag,None)
    if graph in ('buckets','draft','target'): e[FLAGS[0]]='1'
    if graph in ('draft','target'): e[FLAGS[1]]='1'
    if graph=='target': e[FLAGS[2]]='1'
    return e
def note_error(tag,exc):
    errors.append({'tag':tag,'error':str(exc)})
    event('error',tag=tag,error=str(exc))
    save('errors.json',errors)
def bench(tag,group,k=5,adaptive=False,chunk=1024,cublas=False,graph='default',ctx=32768,rep=1):
    guard()
    stem=f'{group}-{tag}-{ctx}-r{rep}'
    path=OUT/(stem+'.json')
    log=OUT/(stem+'.log')
    event('engine_start',tag=stem,k=k,adaptive=adaptive,chunk=chunk,cublas=cublas,graph=graph,context=ctx)
    args=[str(BIN/'ninfer_bench.exe'),'--weights',str(MODEL),'--corpus',str(CORPUS),'--prompt-gen',f'{ctx-1024},512','--max-ctx',str(ctx),'--prefill-chunk',str(chunk),'--kv-dtype','rk4v4','--gdn-state-fp16','--spec','dflash2','--draft-tokens',str(k),'--ngram-draft-tokens','0','--warmup','0','--repetitions','1','--output','json','--output-file',str(path)]
    if adaptive: args+=['--adaptive-mtp']
    if cublas: args+=['--prefill-cublas']
    try:
        with log.open('wb') as f:
            q=sp.run(args,env=environment(graph),cwd=BIN,stdout=f,stderr=sp.STDOUT,timeout=240)
        if q.returncode: raise RuntimeError(f'exit {q.returncode}: {log.name}')
        d=json.loads(path.read_text(encoding='utf-8-sig'))
        t=d['tests'][0]
        if t['reps'][0]['decode_output_tokens']!=512: raise RuntimeError('incomplete decode')
        logtxt=log.read_text(encoding='utf-8',errors='replace')
        row=dict(group=group,tag=tag,k=k,adaptive=adaptive,chunk=chunk,cublas=cublas,graph=graph,ctx=ctx,rep=rep,prefill=t['prefill_tok_s_mean'],decode=t['decode_output_tok_s_mean'],accept=t['speculative']['acceptance_rate'],total_seconds=t['total_seconds_mean'],workspace_peak=t.get('workspace_peak_bytes'),flat_lines=[l.strip() for l in logtxt.splitlines() if re.search(r'flat.*(site|replay|fallback|profile)|bucket.*(graph|profile)',l,re.I)],report=path.name)
        rows.append(row);save('engine-results.json',rows)
        event('engine_done',tag=stem,prefill=round(row['prefill'],2),decode=round(row['decode'],2),accept=round(row['accept'],4),seconds=round(row['total_seconds'],3),flat_lines=row['flat_lines'])
        return row
    except Exception as e: note_error(stem,e)
def select(group,key,eligible=lambda r:True):
    by={}
    for r in rows:
        if r['group']==group and eligible(r): by.setdefault(r['tag'],[]).append(r)
    if not by: return None
    # Prefer variants with both repeats, and use median of the two.
    full={k:v for k,v in by.items() if len(v)>=2} or by
    tag=min(full,key=lambda k:statistics.median(key(r) for r in full[k]))
    return full[tag][0]
def api(path,body=None,timeout=240,token=None):
    data=None if body is None else json.dumps(body,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    headers={'Content-Type':'application/json'}
    if token: headers['Authorization']='Bearer '+token
    req=urllib.request.Request(URL+path,data=data,headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=timeout) as f: return json.load(f)
    except urllib.error.HTTPError as e: raise RuntimeError(f'HTTP {e.code}: '+e.read().decode('utf-8','replace')[:1000]) from e
class Server:
    def __init__(self,tag,k,adaptive,chunk,ngram=15,lanes=2,profile=None,cublas=False,graph='default'):
        self.tag=tag;self.p=None;self.token=uuid.uuid4().hex;self.log=OUT/(tag+'-server.log');self.out=None
        self.args=[str(BIN/'ninfer-serve.exe'),str(MODEL),'--model-id','qwen3.8-27b','--host','127.0.0.1','--port',str(PORT),'--device','0','--max-context','176128','--kv-capacity','auto','--kv-headroom-mib','1024','--kv-dtype','rk4v4','--gdn-state-fp16','--max-concurrency',str(lanes),'--kv-lease-growth','--spec','dflash2','--draft-tokens',str(k),'--ngram-draft-tokens',str(ngram),'--ngram-min-match','12','--prefill-chunk',str(chunk),'--use-alt-prefix-caching','--host-cache-mib','16384','--device-profile','auto','--device-profile-path',str(profile or OUT/'builtin-profile-not-present.json'),'--structured-output','--vision','--vision-residency','overlay','--default-reasoning-effort','xhigh','--recover-invariant-failures','--log-stats-interval-ms','15000','--request-log-jsonl',str(OUT/(tag+'-requests.jsonl')),'--admin-token',self.token]
        if adaptive: self.args+=['--adaptive-mtp']
        if cublas: self.args+=['--prefill-cublas']
        self.env=environment(graph)
    def __enter__(self):
        guard();event('server_start',tag=self.tag)
        self.out=self.log.open('wb')
        self.p=sp.Popen(self.args,cwd=BIN,env=self.env,stdout=self.out,stderr=sp.STDOUT)
        try:
            t=time.monotonic()
            while time.monotonic()-t<110:
                if STOP.exists(): raise InterruptedError('STOP requested')
                if self.p.poll() is not None: raise RuntimeError(f'Server exited {self.p.returncode}')
                try:
                    if any(m['id']=='qwen3.8-27b' for m in api('/v1/models',timeout=2)['data']):
                        event('server_ready',tag=self.tag,seconds=round(time.monotonic()-t,3))
                        return self
                except (OSError,RuntimeError,KeyError): pass
                time.sleep(.5)
            raise RuntimeError('readiness timed out')
        except BaseException:
            self.close()
            raise
    def close(self):
        if self.p and self.p.poll() is None:
            try: api('/shutdown',{},timeout=10,token=self.token)
            except Exception: pass
            try: self.p.wait(timeout=20)
            except sp.TimeoutExpired:
                self.p.terminate()
                try:self.p.wait(timeout=10)
                except sp.TimeoutExpired:self.p.kill();self.p.wait()
        if self.out:self.out.close()
    def __exit__(self,*args):self.close()
def req(tag,kind,body,endpoint='/v1/completions',expected=None):
    if STOP.exists(): raise InterruptedError('STOP requested')
    start=time.monotonic()
    d=api(endpoint,body)
    elapsed=time.monotonic()-start
    usage=d.get('usage',{})
    timing=d.get('timings',{})
    choice=(d.get('choices') or [{}])[0]
    content=choice.get('text') or (choice.get('message') or {}).get('content') or d.get('content') or ''
    prompt=usage.get('prompt_tokens',d.get('tokens_evaluated'))
    out=usage.get('completion_tokens',d.get('tokens_predicted'))
    cache=timing.get('cache_n',d.get('tokens_cached',usage.get('prompt_tokens_details',{}).get('cached_tokens')))
    r={'tag':tag,'kind':kind,'wall_seconds':elapsed,'prompt_tokens':prompt,'output_tokens':out,'cache_tokens':cache,'finish':choice.get('finish_reason',d.get('stop_type')),'timings':timing,'output_sha256':hashlib.sha256(content.encode('utf-8')).hexdigest(),'output_characters':len(content)}
    if expected is not None:
        norm=lambda x:re.sub(r'^\s*'+chr(96)*3+r'(?:\w+)?\s*\n|\n\s*'+chr(96)*3+r'\s*$','',x.strip()).replace('\r\n','\n').strip()
        a,b=norm(content),norm(expected)
        r.update(exact_copy=(a==b),copy_similarity=difflib.SequenceMatcher(None,a,b,autojunk=False).ratio())
    # Keep response locally, not the input/source text in public summaries.
    save(tag+'-response.json',d)
    with lock:
        serv.append(r);save('server-results.json',serv)
        event('server_request_done',**r)
    return r,d
def raw(prompt,n=256,cache=False):
    return {'model':'qwen3.8-27b','prompt':prompt,'max_tokens':n,'temperature':0,'top_k':1,'presence_penalty':0,'frequency_penalty':0,'ignore_eos':True,'cache_prompt':cache}
def chat(code):
    return {'model':'qwen3.8-27b','messages':[{'role':'user','content':'Return this C++ source file verbatim, but replace the exact identifier BENCHMARK_ORIGINAL with BENCHMARK_RENAMED wherever it occurs. Output only the source, without Markdown or explanations.\n\n'+code}],'max_tokens':1800,'temperature':0,'top_k':1,'reasoning_effort':'none','cache_prompt':False}
def server_case(tag,fn,**kwargs):
    try:
        with Server(tag,**kwargs) as s:fn(s)
    except InterruptedError:raise
    except Exception as e:note_error(tag,e)
def main():
    guard()
    ids=[int(x) for x in CORPUS_SRC.read_text().split()]
    ids=(ids*((176128+len(ids)-1)//len(ids)))[:176128]
    CORPUS.write_bytes((' '.join(map(str,ids))+'\r\n').encode('ascii'))
    version=sp.check_output([str(BIN/'ninfer-serve.exe'),'--version'],cwd=BIN,text=True,stderr=sp.STDOUT).strip()
    if not ALLOW_OTHER_BUILD and 'rtx4090-v0.1.4+a6b1138' not in version: raise RuntimeError(version)
    save('manifest.json',{'started':stamp(),'version':version,'server_sha256':digest(BIN/'ninfer-serve.exe'),'bench_sha256':digest(BIN/'ninfer_bench.exe'),'runtime_sha256':digest(BIN/'backends'/'cuda13'/'ninfer-runtime.dll'),'model_sha256':digest(MODEL),'corpus_sha256':digest(CORPUS),'corpus_tokens':len(ids),'notes':'Synthetic repeated fixed token corpus for controlled engine tests; separate real repository file-copy workload for serving. Two single-run reverse-order screens; no speedup or general quality claims from synthetic tokens. No production changes.'})
    event('manifest_ready')
    # Screen all 5 widths and both adaptive settings. Repeat in reversed order.
    draft=[(f'k{k}-'+('adaptive' if a else 'fixed'),dict(k=k,adaptive=a)) for k in (5,7,9,11,15) for a in (False,True)]
    for rep in (1,2):
        for tag,kw in (draft if rep==1 else list(reversed(draft))):bench(tag,'draft',rep=rep,**kw)
    best=select('draft',lambda r:-r['decode']) or {'k':5,'adaptive':False}
    bk,ba=best['k'],best['adaptive']
    event('draft_screen_done',winner_k=bk,adaptive=ba)
    # Prefill all 3 x 2, using selected width; compare cold prompt and total cost.
    pref=[(f'p{p}-'+('cublas' if c else 'native'),dict(k=bk,adaptive=ba,chunk=p,cublas=c)) for p in (1024,2048,4096) for c in (False,True)]
    for rep in (1,2):
        for tag,kw in (pref if rep==1 else list(reversed(pref))):bench(tag,'prefill',rep=rep,**kw)
    bp=(select('prefill',lambda r:r['total_seconds'],lambda r:not r['cublas']) or {'chunk':1024})['chunk']
    event('prefill_screen_done',native_chunk=bp)
    # Flat graph isolated screen. These remain experimental regardless of speed.
    graphs=['default','buckets','draft','target']
    for rep in (1,2):
        for g in (graphs if rep==1 else list(reversed(graphs))):bench(g,'graph',k=bk,adaptive=ba,chunk=bp,graph=g,rep=rep)
    # Long-context baseline versus selected fixed/adaptive width/native prefill.
    for ctx in (131072,176128):
        arms=[('baseline',dict(k=5,adaptive=False,chunk=1024)),('native-candidate',dict(k=bk,adaptive=ba,chunk=bp))]
        for rep in (1,2):
            for tag,kw in (arms if rep==1 else list(reversed(arms))):bench(tag,'long',ctx=ctx,rep=rep,**kw)
    # Real repository source used for exact-copy/refactor workload (no private files).
    paths=['src/artifact/views.cpp','src/core/expert_mirror.cpp','src/core/expert_split.cpp']
    # Original short fixtures came from the local 4cca384f checkout.
    files=[sp.check_output(['git','-C',str(SOURCE),'show','4cca384f:'+p]) for p in paths]
    samples=['// BENCHMARK_ORIGINAL\n'+d.decode('utf-8').replace('\r\n','\n') for d in files]
    save('coding-fixtures.json',[{'path':p,'sha256':hashlib.sha256(d.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest()} for p,d in zip(paths,files)])
    opts=dict(k=bk,adaptive=ba,chunk=bp)
    def coding(s):
        for i,code in enumerate(samples):
            req(f'{s.tag}-copy{i}','coding',chat(code),'/v1/chat/completions',code.replace('BENCHMARK_ORIGINAL','BENCHMARK_RENAMED'))
    # N-gram test: same 3 real files with and without copy proposals.
    for n in (0,15,15,0):
        i=len([r for r in serv if r['kind']=='coding'])//3+1
        server_case(f'ngram{n}-r{i}',coding,ngram=n,lanes=2,**opts)
    # Concurrency: same pair launched simultaneously; C1 queues vs C2 interleaves.
    # Raw exact length ensures same output-token budget, independent of EOS.
    def concurrency(s):
        prompts=[ids[:31744],ids[1024:32768]]
        start=time.monotonic()
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            futures=[ex.submit(req,f'{s.tag}-q{i}','concurrency',raw(p,512)) for i,p in enumerate(prompts)]
            rs=[f.result()[0] for f in futures]
        elapsed=time.monotonic()-start
        event('concurrency_pair_done',tag=s.tag,wall_seconds=elapsed,aggregate_tokens_s=sum(r['output_tokens'] or 0 for r in rs)/elapsed)
        save(s.tag+'-pair.json',{'wall_seconds':elapsed,'requests':rs,'aggregate_tokens_s':sum(r['output_tokens'] or 0 for r in rs)/elapsed})
    for i,c in enumerate((1,2,2,1),1):server_case(f'concurrency{c}-r{i}',concurrency,lanes=c,**opts)
    # Compare identical server request with built-in and local calibrated routes.
    def profile_case(s):
        for i in range(2):req(f'{s.tag}-cold{i}','profile',raw(ids[:31744],512))
    server_case('profile-builtin',profile_case,**opts)
    guard()
    prof=OUT/'rtx4090-sm89-v014-calibrated.json'
    event('calibration_start')
    try:
        with (OUT/'calibrate.log').open('wb') as f:
            p=sp.run([str(BIN/'ninfer-calibrate.exe'),'--device','0','--out',str(prof),'--repeat','3','--detail'],cwd=BIN,env=environment(),stdout=f,stderr=sp.STDOUT,timeout=600)
        if p.returncode or not prof.exists():raise RuntimeError(f'calibration exit {p.returncode}')
        event('calibration_done',profile=str(prof),sha256=digest(prof))
        server_case('profile-calibrated',profile_case,profile=prof,**opts)
    except Exception as e:note_error('calibration',e)
    # Long context cache: identical regeneration and actual continuation of output.
    # /completion returns token IDs; use them in the subsequent request.
    def cache_case(s):
        prefix=ids[:175104]
        body=raw(prefix,128,True);body['return_tokens']=True
        a,first=req(f'{s.tag}-cold','cache176k',body,'/completion')
        b,second=req(f'{s.tag}-repeat','cache176k',body,'/completion')
        tokens=second.get('tokens')
        if isinstance(tokens,list) and all(isinstance(t,int) for t in tokens) and tokens:
            q=prefix+tokens+ids[-32:]
            cont=raw(q,128,True);cont['return_tokens']=True
            req(f'{s.tag}-continue','cache176k',cont,'/completion')
        else:
            q=prefix+ids[-32:]
            cont=raw(q,128,True);cont['return_tokens']=True
            req(f'{s.tag}-extended','cache176k',cont,'/completion')
    server_case('hybrid176k-baseline',cache_case,k=5,adaptive=False,chunk=1024,lanes=2)
    server_case('hybrid176k-candidate',cache_case,lanes=2,**opts)
    # Test cuBLAS speed on coding correctness as a distinct tradeoff.
    bc=select('prefill',lambda r:r['total_seconds'],lambda r:r['cublas'])
    if bc:server_case('cublas-coding',coding,k=bk,adaptive=ba,chunk=bc['chunk'],cublas=True)
    save('summary.json',{'status':'finished','finished':stamp(),'engine_runs':len(rows),'server_requests':len(serv),'errors':errors,'screen_candidate':{'draft_tokens':bk,'adaptive':ba,'native_prefill_chunk':bp},'note':'Screen candidate only; compare long context and serving results, variance, quality, concurrency and cache before changing production. cuBLAS and graph experiments are not automatically enabled.'})
    event('finished',engine_runs=len(rows),server_requests=len(serv),errors=len(errors))
if __name__=='__main__':
    cli()
    tele=None;f=None
    try:
        guard()
        f=(OUT/'gpu-telemetry.csv').open('wb')
        tele=sp.Popen(['nvidia-smi','--query-gpu=timestamp,name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.free,utilization.gpu,utilization.memory','--format=csv','--loop-ms=500'],stdout=f,stderr=sp.STDOUT)
        main()
    except BaseException as e:
        note_error('campaign',e)
        save('summary.json',{'status':'stopped' if isinstance(e,InterruptedError) else 'failed','finished':stamp(),'engine_runs':len(rows),'server_requests':len(serv),'errors':errors})
        raise
    finally:
        if tele and tele.poll() is None:tele.terminate();tele.wait(timeout=10)
        if f:f.close()
