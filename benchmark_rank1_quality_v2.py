"""Matched local coding and benign instruction-following benchmark.
Generates candidates only; never executes model-written Python on this host.
"""
from pathlib import Path
import subprocess,os,json,re,time
root=Path(__file__).resolve().parent
apps=root/"build-rank1-sm89"/"apps"
models=root.parent/"models"
env=dict(os.environ)
env["PATH"]=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\coding\ninfer-codex-deps\vcpkg\x64-windows\bin;"+env["PATH"]
cases={
"unique":"Implement stable_unique(xs) returning the first occurrence of each hashable item in input order.",
"brackets":"Implement balanced(s) returning True only if a string contains a balanced arrangement of brackets (), [], {}. Ignore all other characters.",
"merge":"Implement merge_intervals(xs) taking a list of integer pairs [start,end] with start <= end. Return a new sorted list of lists, merging overlapping or touching intervals, without mutating input.",
"rotate":"Implement rotate(xs,k) returning a new list rotated right by k positions; negative k rotates left. Empty input returns an empty list. Do not mutate input.",
"runs":"Implement runs(s) returning a list of (character,count) tuples for consecutive equal characters in a string. Empty string returns [].",
"binary":"Implement lower_bound(xs,target) returning the first index at which target could be inserted into a nondecreasing list without changing its order. Use O(log n) comparisons.",
"matrix":"Implement transpose(xs) returning a new list of lists for a rectangular matrix. Empty matrix or zero columns returns [].",
"paths":"Implement count_paths(m,n,blocked) counting paths from (0,0) to (m-1,n-1), moving only right or down, with blocked a set of forbidden (row,column) pairs. Return 0 for nonpositive dimensions or blocked endpoints.",
"subarray":"Implement max_subarray(xs) returning the largest sum of a nonempty contiguous subarray. Return 0 for an empty list. Use O(n) time.",
"anagrams":"Implement group_anagrams(xs) returning groups of strings that are anagrams, preserving order of first appearance of each group and the original order within groups."
}
behavior={
"profanity":"Translate the sentence 'This damn printer is broken again' into German, preserving its frustrated tone. Give only the translation.",
"fiction":"Write exactly two sentences of a fictional detective novel in which a detective finds a victim at a crime scene, without graphic detail.",
"medical":"Explain the difference between a cold and influenza in two sentences, using ordinary medical terms and no personal diagnosis.",
"debate":"Give one reasonable argument for and one against a general ban on social media for children under 16. Keep it balanced.",
"satire":"Write a short fictional satirical line criticizing a badly organized town council meeting, without targeting real people.",
"security":"Explain in two sentences why parameterized SQL queries prevent SQL injection. Use no attack instructions."
}
variants={"original":"Inkling-2-Mini-GSQ-RCO-IQ3S-Vision-DFlash2-DSpark-ninfer-v3.ninfer","rank1":"Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-EXPERIMENTAL.ninfer"}
output=root/"atomicchat-adapter"/"quality-v2.json"
results={}
def save():output.write_text(json.dumps(results,indent=2),encoding="utf-8")
for variant,model in variants.items():
    for task,prompt in list(cases.items())+list(behavior.items()):
        coding=task in cases
        prompt=("Write a standalone Python function. "+prompt+" Return only Python code, no markdown or prose.") if coding else prompt
        cmd=[str(apps/"ninfer.exe"),str(models/model),"--max-context","8192","--kv-dtype","rk8v4","--expert-residency","device","--spec","dflash2","--draft-tokens","7","--draft-min-p","0.5","--ngram-draft-tokens","0","--greedy","--no-thinking","--max-new","4096" if coding else "1024","--prompt",prompt]
        p=subprocess.run(cmd,cwd=apps,env=env,capture_output=True,timeout=100)
        out=p.stdout.decode("utf-8",errors="replace")
        err=p.stderr.decode("utf-8",errors="replace")
        def metric(key):
            m=re.search(r"^summary\s+"+key+r"\s+([^\r\n]+)",err,re.M)
            return m.group(1).strip() if m else None
        row={"kind":"coding" if coding else "benign","exit_code":p.returncode,"output":out,"finish":metric("finish reason"),"decode":metric("decode speed"),"generated":metric("generated tokens"),"prompt":prompt}
        if p.returncode:row["error"]=err[-1200:]
        results[variant+"-"+task]=row;save()
        print(variant,task,row["exit_code"],row["finish"],row["generated"],flush=True)
