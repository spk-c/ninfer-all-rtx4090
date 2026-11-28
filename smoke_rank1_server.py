from pathlib import Path
import subprocess,os,json,time,urllib.request,urllib.error
root=Path(__file__).resolve().parent
script=root.parent/"Start-Inkling2-GSQ-RCO-Rank1-RTX4090-Experimental.ps1"
logdir=root/"atomicchat-adapter"
port=8082
url=f"http://127.0.0.1:{port}"
def call(path,data=None,timeout=60):
    r=urllib.request.Request(url+path,data=None if data is None else json.dumps(data).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=timeout) as f:return json.load(f)
try:
    call("/v1/models",timeout=1)
except (OSError,urllib.error.URLError):pass
else:raise RuntimeError("Port already occupied")
results={}
with (logdir/"server-smoke.log").open("wb") as log:
    p=subprocess.Popen(["powershell","-NoProfile","-ExecutionPolicy","Bypass","-File",str(script),"-Port",str(port)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
    try:
        for i in range(60):
            if p.poll() is not None:raise RuntimeError("Start script exited "+str(p.returncode))
            try:
                results["models"]=call("/v1/models",timeout=2);break
            except (OSError,urllib.error.URLError):time.sleep(1)
        else:raise RuntimeError("Server startup timeout")
        model="inkling2-gsq-rco-rank1-experimental"
        common={"model":model,"temperature":0,"max_tokens":1024,"stream":False}
        results["chat"]=call("/v1/chat/completions",{**common,"messages":[{"role":"user","content":"Return the result of 19 + 23. Be concise."}]})
        tool={"type":"function","function":{"name":"lookup_weather","description":"Look up weather in a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":False}}}
        results["tool"]=call("/v1/chat/completions",{**common,"messages":[{"role":"user","content":"Call lookup_weather for Berlin. Do not invent the weather."}],"tools":[tool],"tool_choice":{"type":"function","function":{"name":"lookup_weather"}}})
        msg=results["tool"]["choices"][0]["message"]
        calls=msg.get("tool_calls",[])
        results["tool_valid"]=any(c["function"]["name"]=="lookup_weather" and json.loads(c["function"]["arguments"]).get("city")=="Berlin" for c in calls)
        print("SERVER",json.dumps(results,ensure_ascii=True),flush=True)
    finally:
        if p.poll() is None:subprocess.run(["taskkill","/PID",str(p.pid),"/T","/F"],capture_output=True,timeout=20)
        (logdir/"server-smoke.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
