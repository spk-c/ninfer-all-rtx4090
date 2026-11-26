"""Audit public external links. Hard failures are reported separately from transient blocks."""
from html.parser import HTMLParser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag
import json, sys

BASE="https://spk-c.github.io/ninfer-all-rtx4090/"
class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.urls=set()
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        u=a.get("href","") if tag=="a" else ""
        if u.startswith("https://") and not u.startswith(BASE):
            self.urls.add(urldefrag(u)[0])
links=Links()
for p in Path(sys.argv[1] if len(sys.argv)>1 else "site").glob("*.html"):
    links.feed(p.read_text(encoding="utf-8"))
def probe(u):
    try:
        req=Request(u,headers={"User-Agent":"NInfer-Website-Link-Audit/1.0","Accept":"text/html,*/*"},method="HEAD")
        try:
            with urlopen(req,timeout=12) as res:return (u,res.status,"ok")
        except HTTPError as e:
            if e.code not in (403,405,404,410):raise
        req=Request(u,headers={"User-Agent":"NInfer-Website-Link-Audit/1.0","Range":"bytes=0-2047"})
        with urlopen(req,timeout=12) as res:return (u,res.status,"ok")
    except HTTPError as e:return (u,e.code,"broken" if e.code in (404,410) else "unverified")
    except (URLError,TimeoutError,OSError) as e:return (u,0,"unverified")
with ThreadPoolExecutor(max_workers=5) as pool:
    results=list(pool.map(probe,sorted(links.urls)))
for row in results:print(*row)
print(json.dumps({status:sum(r[2]==status for r in results) for status in ("ok","broken","unverified")}))
if any(r[2]=="broken" for r in results):sys.exit(1)
