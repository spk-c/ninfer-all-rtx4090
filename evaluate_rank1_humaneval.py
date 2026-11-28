"""Functional HumanEval scoring in a disposable, offline Linux container only.
Run with no network, read-only inputs, non-root UID, no capabilities, bounded
memory/processes and no host mounts except the benchmark inputs.
Never run this evaluator on the host or a system holding secrets.
"""
from pathlib import Path
import ast, gzip, json, os, re, subprocess, sys, tempfile, time
from collections import Counter
import resource
root=Path("/input")
if os.name!="posix" or os.getuid()==0 or not Path("/.dockerenv").exists():
    raise SystemExit("Use the disposable non-root Docker runner, not the host")
with gzip.open(root/"HumanEval.jsonl.gz","rt",encoding="utf-8") as f:
    problems={p["task_id"]:p for p in map(json.loads,f)}
assert len(problems)==164
def limit():
    resource.setrlimit(resource.RLIMIT_CPU,(4,5))
    resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,512*1024*1024))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
    resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
def grade(problem,candidate):
    s=candidate.strip()
    match=re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```",s,re.S|re.I)
    if match:s=match.group(1)
    try:ast.parse(s)
    except SyntaxError:return "syntax"
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        path=Path(tmp)/"candidate.py"
        path.write_text(s+"\n\n"+problem["test"]+"\n\ncheck("+problem["entry_point"]+")\n",encoding="utf-8")
        try:
            p=subprocess.run([sys.executable,"-I","-B",str(path)],cwd=tmp,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env={"PATH":"/usr/local/bin:/usr/bin:/bin","PYTHONHASHSEED":"0"},timeout=8,preexec_fn=limit,start_new_session=True)
            return "pass" if p.returncode==0 else "fail"
        except subprocess.TimeoutExpired:return "timeout"
results={}
for variant in ("original","rank1"):
    rows=[json.loads(line) for line in (root/(variant+".jsonl")).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows)==len(problems)==len({r["task_id"] for r in rows}) and {r["task_id"] for r in rows}==set(problems)
    grades={}
    for row in rows:
        key=row["task_id"];grades[key]=grade(problems[key],row["candidate"]) if not row.get("error") else "generation_error"
    summary=dict(Counter(grades.values()))
    results[variant]={"summary":summary,"pass_at_1":summary.get("pass",0)/164,"tasks":grades}
print(json.dumps(results,indent=2))
