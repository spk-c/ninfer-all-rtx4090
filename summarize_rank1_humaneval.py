from pathlib import Path
import json, statistics, ast, re
root=Path(__file__).resolve().parent
folder=root/"atomicchat-adapter"/"humaneval"
out={}
for name in ("original","rank1"):
    rows=[json.loads(line) for line in (folder/(name+".jsonl")).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows)==164 and len({r["task_id"] for r in rows})==164
    valid=0;bad=[];empty=[];rates=[];tokens=[];finish={};errors=[]
    for r in rows:
        f=r.get("finish","error");finish[f]=finish.get(f,0)+1
        if r.get("error"):errors.append(r["task_id"])
        s=r.get("candidate","").strip()
        match=re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```",s,re.S|re.I)
        if match:s=match.group(1)
        if not s:empty.append(r["task_id"])
        try:
            t=ast.parse(s)
            if any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==r["entry_point"] for n in t.body):valid+=1
            else:bad.append(r["task_id"])
        except SyntaxError:bad.append(r["task_id"])
        speed=(r.get("timings") or {}).get("predicted_per_second")
        if isinstance(speed,(float,int)):rates.append(speed)
        n=(r.get("usage") or {}).get("completion_tokens")
        if isinstance(n,int):tokens.append(n)
    out[name]={"rows":len(rows),"finish":finish,"errors":errors,"empty":empty,"syntax_and_entry_point":valid,"syntax_or_entry_point_missing":bad,"mean_decode":statistics.mean(rates) if rates else None,"median_decode":statistics.median(rates) if rates else None,"total_generated_tokens":sum(tokens),"mean_generated_tokens":statistics.mean(tokens) if tokens else None}
(folder/"static-summary.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
