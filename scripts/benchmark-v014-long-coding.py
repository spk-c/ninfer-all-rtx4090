from pathlib import Path
import concurrent.futures as cf
import hashlib, importlib.util, json, os, re, statistics, subprocess as sp, threading, time, urllib.request, urllib.error

import argparse
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('base_campaign',HERE/'benchmark-v014-tuning.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
PREV=OUT=ids=None
def cli():
    global PREV,OUT,ids
    p=argparse.ArgumentParser(description='Reproduce RTX4090 v0.1.4 long coding campaign')
    p.add_argument('--bin-dir',required=True,type=Path);p.add_argument('--model',required=True,type=Path)
    p.add_argument('--repo',default=HERE.parent,type=Path);p.add_argument('--previous',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--port',type=int,default=8099)
    p.add_argument('--allow-other-build',action='store_true',help='Explicitly allow a later release for regression screening')
    a=p.parse_args();PREV=a.previous.resolve();OUT=a.out.resolve()
    b.configure(a.bin_dir,a.model,a.repo,OUT,None,a.port,a.allow_other_build)
    b.CORPUS=PREV/'long-context-176k.ids'
    ids=[int(x) for x in b.CORPUS.read_text().split()]
    if len(ids)!=176128:raise RuntimeError('Unexpected corpus length')
CONTEXTS=(65536,131072,176128)
KS=(5,7,9,11,15)
buckets=[]
overlap=[]
ranking=[]
winner=None

def git_source(path):
    return sp.check_output(['git','-C',str(b.SOURCE),'show','a6b1138:'+path]).decode('utf-8')
def rank():
    groups={}
    for x in b.rows:
        if x['group']=='draft-long':groups.setdefault((x['ctx'],x['k']),[]).append(x)
    results=[]
    for ctx in CONTEXTS:
        for k in KS:
            a=groups.get((ctx,k),[])
            if len(a)!=2:raise RuntimeError(f'Incomplete draft measurement: {ctx}/{k}: {len(a)}')
            results.append({'ctx':ctx,'k':k,'decode':statistics.median(x['decode'] for x in a),'prefill':statistics.median(x['prefill'] for x in a),'seconds':statistics.median(x['total_seconds'] for x in a),'acceptance':statistics.median(x['accept'] for x in a),'decode_min':min(x['decode'] for x in a),'decode_max':max(x['decode'] for x in a)})
    maxima={ctx:max(x['decode'] for x in results if x['ctx']==ctx) for ctx in CONTEXTS}
    scores=[]
    weights={65536:.25,131072:.35,176128:.40}
    for k in KS:
        by=[x for x in results if x['k']==k]
        scores.append({'k':k,'weighted_decode_score':sum(weights[x['ctx']]*x['decode']/maxima[x['ctx']] for x in by),'worst_relative_decode':min(x['decode']/maxima[x['ctx']] for x in by)})
    scores.sort(key=lambda x:x['weighted_decode_score'],reverse=True)
    b.save('draft-ranking.json',{'results':results,'scores':scores,'weights':weights,'rule':'Median of two reverse-order measurements per cell, relative to fastest width at same context; weights prioritize long-context workloads. Synthetic corpus, N-gram off.'})
    return scores
def coding_fixtures():
    targets=[('src/serve/request_validation.cpp','optional_number','optional_finite_number'),('src/serve/http_transport.cpp','set_ngram_generation_header','publish_ngram_generation_header'),('src/serve/structured_output.cpp','parse_structured_output','parse_response_format_options')]
    refs=['src/serve/openai_common.cpp','src/serve/openai_chat_http.cpp','src/serve/openai_responses_http.cpp','src/serve/request_validation.cpp','src/serve/http_transport.cpp','src/serve/structured_output.cpp']
    src={p:git_source(p) for p in refs}
    fixtures=[]
    manifest=[]
    for path,old,new in targets:
        target=src[path]; count=len(re.findall(r'\b'+re.escape(old)+r'\b',target))
        if count<1 or re.search(r'\b'+re.escape(new)+r'\b',target):raise RuntimeError(f'Invalid rename fixture: {path}')
        expected=re.sub(r'\b'+re.escape(old)+r'\b',new,target)
        references='\n\n'.join('REFERENCE FILE: '+p+'\n'+src[p] for p in refs if p!=path)
        prompt=('You are editing a C++ repository. The following reference files are context only. Do not output or modify them.\n\n'+references+
                '\n\nTASK: In the target file below, rename only the exact C++ identifier '+old+' to '+new+
                ' wherever it occurs. Preserve every other character, including comments and formatting. Return the entire updated target file only, with no Markdown, no explanations and no omitted sections.\nTARGET FILE: '+path+'\n'+target)
        body={'model':'qwen3.8-27b','messages':[{'role':'user','content':prompt}],'max_tokens':8192,'temperature':0,'top_k':1,'reasoning_effort':'none','cache_prompt':False}
        fixtures.append((path,body,expected))
        manifest.append({'path':path,'old':old,'new':new,'occurrences':count,'target_bytes':len(target.encode()),'prompt_bytes':len(prompt.encode()),'target_sha256':hashlib.sha256(target.encode()).hexdigest(),'expected_sha256':hashlib.sha256(expected.encode()).hexdigest(),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
    b.save('coding-fixtures.json',manifest)
    return fixtures

def raw_stream(tag,prompt,n,started=None):
    if b.STOP.exists():raise InterruptedError('STOP requested')
    body=b.raw(prompt,n,False)
    body.update(stream=True,stream_options={'include_usage':True})
    start=time.monotonic();times=[];parts=[];usage=None;timings=None;finish=None
    req=urllib.request.Request(b.URL+'/v1/completions',data=json.dumps(body,separators=(',',':')).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=240) as res:
            for line in res:
                if b.STOP.exists():raise InterruptedError('STOP requested')
                line=line.decode('utf-8','replace').strip()
                if not line.startswith('data:'):continue
                payload=line[5:].strip()
                if payload=='[DONE]':break
                e=json.loads(payload)
                if e.get('error'):raise RuntimeError(str(e['error']))
                if e.get('usage'):usage=e['usage']
                if e.get('timings'):timings=e['timings']
                for c in e.get('choices') or []:
                    text=c.get('text') or ''
                    if text:
                        times.append(time.monotonic());parts.append(text)
                        if started:started.set()
                    if c.get('finish_reason'):finish=c['finish_reason']
    except urllib.error.HTTPError as e:raise RuntimeError(f'HTTP {e.code}: '+e.read().decode('utf-8','replace')[:1000]) from e
    end=time.monotonic()
    if not times or not usage or usage.get('completion_tokens')!=n:raise RuntimeError(f'Incomplete streaming response {tag}: usage={usage}, finish={finish}')
    gaps=[(t-s)*1000 for s,t in zip(times,times[1:])]
    sg=sorted(gaps)
    r={'tag':tag,'prompt_tokens':usage.get('prompt_tokens'),'output_tokens':usage.get('completion_tokens'),'wall_seconds':end-start,'ttft_seconds':times[0]-start,'decode_chunk_p95_gap_ms':sg[min(len(sg)-1,int(.95*len(sg)))] if sg else None,'decode_chunk_max_gap_ms':max(gaps) if gaps else None,'stream_chunks':len(times),'output_sha256':hashlib.sha256(''.join(parts).encode()).hexdigest(),'finish':finish,'timings':timings,'start_monotonic':start,'first_chunk_monotonic':times[0],'last_chunk_monotonic':times[-1],'end_monotonic':end}
    b.save(tag+'-stream.json',r)
    return r
def pair(tag,mode):
    # The long request is first admitted, then a short request arrives during
    # its prefill. A second scenario introduces a long prompt while a short
    # prompt is already streaming. Record actual observed overlap.
    t=time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        if mode=='prefill':
            a=ex.submit(raw_stream,tag+'-long',ids[:64512],512)
            time.sleep(1.0)
            c=ex.submit(raw_stream,tag+'-short',ids[70000:74096],512)
        else:
            first=threading.Event()
            a=ex.submit(raw_stream,tag+'-first',ids[70000:78192],4096,first)
            if not first.wait(45):raise RuntimeError('No initial streaming chunk for decode overlap')
            c=ex.submit(raw_stream,tag+'-second',ids[:31744],512)
        r1,r2=a.result(),c.result()
    elapsed=time.monotonic()-t
    if mode=='prefill':
        observed=r1['start_monotonic']<r2['start_monotonic']<r1['first_chunk_monotonic']
    else:
        observed=r1['first_chunk_monotonic']<=r2['start_monotonic']<r1['last_chunk_monotonic']
    r={'tag':tag,'scenario':mode,'offered_overlap_confirmed':observed,'wall_seconds':elapsed,'aggregate_output_tokens_s':(r1['output_tokens']+r2['output_tokens'])/elapsed,'requests':[r1,r2]}
    overlap.append(r);b.save('concurrency-results.json',overlap)
    b.event('overlap_pair_done',tag=tag,scenario=mode,offered_overlap=observed,first_ttft=r1['ttft_seconds'],second_ttft=r2['ttft_seconds'],first_max_gap_ms=r1['decode_chunk_max_gap_ms'],second_max_gap_ms=r2['decode_chunk_max_gap_ms'],wall_seconds=elapsed)

def main():
    global winner,ranking
    b.guard()
    version=sp.check_output([str(b.BIN/'ninfer-serve.exe'),'--version'],cwd=b.BIN,text=True,stderr=sp.STDOUT).strip()
    if not b.ALLOW_OTHER_BUILD and 'rtx4090-v0.1.4+a6b1138' not in version:raise RuntimeError(version)
    prev=json.loads((PREV/'manifest.json').read_text(encoding='utf-8-sig'))
    hashes={'server_sha256':b.digest(b.BIN/'ninfer-serve.exe'),'bench_sha256':b.digest(b.BIN/'ninfer_bench.exe'),'runtime_sha256':b.digest(b.BIN/'backends'/'cuda13'/'ninfer-runtime.dll'),'model_sha256':b.digest(b.MODEL),'corpus_sha256':b.digest(b.CORPUS)}
    for k,v in hashes.items():
        if prev.get(k)!=v:raise RuntimeError('Previous run identity mismatch: '+k)
    b.save('manifest.json',{'started':b.stamp(),'version':version,**hashes,'contexts':CONTEXTS,'draft_tokens':KS,'prefill_chunk':4096,'repetitions':2,'draft_screen_ngram':0,'adaptive':False,'cublas':False,'notes':'30 engine runs, 4 bucket server runs, 12 longer real-source coding requests, 8 overlapping pairs. Same exact model/binary/corpus as prior campaign. Source fixtures from release commit a6b1138. All server traffic localhost. No production config changes.'})
    fixtures=coding_fixtures()
    b.event('manifest_ready')
    # Reversed ordering of both context and width on repetition two.
    for rep in (1,2):
        for ctx in (CONTEXTS if rep==1 else tuple(reversed(CONTEXTS))):
            for k in (KS if rep==1 else tuple(reversed(KS))):
                b.bench(f'k{k}','draft-long',k=k,adaptive=False,chunk=4096,ctx=ctx,rep=rep)
    ranking=rank();winner=ranking[0]['k']
    b.event('long_draft_screen_done',winner_k=winner,ranking=ranking)
    opts=dict(k=winner,adaptive=False,chunk=4096,lanes=2)
    # Same exact large prompt and output budget; N-gram off isolates bucket routing.
    for i,graph in enumerate(('default','buckets','buckets','default'),1):
        tag=f'buckets176k-{graph}-r{i}'
        def test(s):
            body=b.raw(ids[:175104],512,False);body['return_tokens']=True
            r,d=b.req(s.tag+'-cold','buckets176k',body,'/completion')
            tokens=d.get('tokens')
            if r['prompt_tokens']!=175104 or r['output_tokens']!=512 or not isinstance(tokens,list) or len(tokens)!=512:
                raise RuntimeError('Incomplete 176K bucket comparison')
            r=dict(r);r['graph']=graph;r['token_ids_sha256']=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()
            buckets.append(r);b.save('buckets-results.json',buckets)
        b.server_case(tag,test,ngram=0,graph=graph,**opts)
    # Long realistic multi-file context and complete target file replacement.
    for i,n in enumerate((0,15,15,0),1):
        def coding(s):
            for q,(path,body,expected) in enumerate(fixtures):
                r,d=b.req(s.tag+f'-rename{q}','coding-long',body,'/v1/chat/completions',expected)
                if r['finish']!='stop':b.event('coding_quality_warning',tag=r['tag'],finish=r['finish'],exact=r.get('exact_copy'))
        b.server_case(f'long-ngram{n}-r{i}',coding,ngram=n,**opts)
    # ABBA comparison on a real HTTP server with a streaming latency trace.
    # The only changed server argument is --concurrent-prefill.
    for i,enabled in enumerate((False,True,True,False),1):
        tag=f'concurrent-prefill-{int(enabled)}-r{i}'
        try:
            server=b.Server(tag,ngram=15,**opts)
            if enabled:server.args+=['--concurrent-prefill']
            with server:
                pair(tag+'-prefill','prefill')
                pair(tag+'-decode','decode')
        except InterruptedError:raise
        except Exception as e:b.note_error(tag,e)
    summary('finished' if not b.errors else 'finished-with-errors')
    b.event('finished',engine_runs=len(b.rows),server_requests=len(b.serv),overlap_pairs=len(overlap),errors=len(b.errors),winner_k=winner)
def summary(status):
    b.save('summary.json',{'status':status,'finished':b.stamp(),'engine_runs':len(b.rows),'server_nonstream_requests':len(b.serv),'overlap_pairs':len(overlap),'stream_requests':2*len(overlap),'bucket_runs':len(buckets),'errors':b.errors,'draft_candidate':winner,'ranking':ranking,'note':'Synthetic K screen; use long coding output correctness, bucket equivalence, observed overlap and latency results before changing production. No automatic changes.'})
if __name__=='__main__':
    cli()
    tele=None;f=None
    try:
        b.guard()
        f=(OUT/'gpu-telemetry.csv').open('wb')
        tele=sp.Popen(['nvidia-smi','--query-gpu=timestamp,name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.free,utilization.gpu,utilization.memory','--format=csv','--loop-ms=500'],stdout=f,stderr=sp.STDOUT)
        main()
    except BaseException as e:
        b.note_error('campaign',e);summary('stopped' if isinstance(e,InterruptedError) else 'failed');raise
    finally:
        if tele and tele.poll() is None:tele.terminate();tele.wait(timeout=10)
        if f:f.close()
