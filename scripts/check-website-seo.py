"""Validate static project portal; no network required."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
import re
import sys
import xml.etree.ElementTree as ET

BASE = "https://spk-c.github.io/ninfer-all-rtx4090/"
ROOT = Path(sys.argv[1] if len(sys.argv)>1 else "site")
class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.h1 = 0
        self.meta = {}
        self.canonical = []
        self.alternates = {}
        self.language = None
        self.links = []
        self.ids = []
        self.title_count = 0
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html": self.language = a.get("lang")
        if tag == "link" and a.get("rel") == "alternate" and a.get("hreflang"):
            self.alternates[a["hreflang"]] = a.get("href")
        if tag == "h1": self.h1 += 1
        if tag == "title": self.title_count += 1
        if tag == "meta":
            k = a.get("name") or a.get("property")
            if k: self.meta[k] = a.get("content", "")
        if tag == "link" and a.get("rel") == "canonical":
            self.canonical.append(a.get("href"))
        if "id" in a: self.ids.append(a["id"])
        if tag in ("a", "link", "script", "img"):
            u = a.get("href") or a.get("src")
            if u: self.links.append(u)

pages = {}
errors = []
def check(ok, message):
    if not ok: errors.append(message)

for path in sorted(ROOT.glob("*.html")):
    parser = Page()
    parser.feed(path.read_text(encoding="utf-8"))
    pages[path.name] = parser
    check(parser.h1 == 1, f"{path.name}: expected one h1")
    check(parser.title_count == 1, f"{path.name}: expected one title")
    check(len(parser.ids) == len(set(parser.ids)), f"{path.name}: duplicate IDs")
    if path.name == "404.html":
        check("noindex" in parser.meta.get("robots", ""), "404 must be noindex")
        continue
    expected = BASE + ("" if path.name == "index.html" else path.name)
    check(parser.canonical == [expected], f"{path.name}: canonical mismatch")
    check(bool(parser.meta.get("description")), f"{path.name}: missing description")
    check("noindex" not in parser.meta.get("robots",""), f"{path.name}: blocked")
    expected_image = "monitor-desktop.png" if path.name in ("monitor.html","monitor-de.html") else "social-preview.png"
    check(parser.meta.get("og:image") == BASE+expected_image, f"{path.name}: missing social image")
    check(parser.meta.get("twitter:card") == "summary_large_image", f"{path.name}: incorrect social card")

for name,p in pages.items():
    for raw in p.links:
        u = urlsplit(raw)
        if raw.startswith(BASE):
            raw = raw[len(BASE):]
            u = urlsplit(raw)
        elif u.scheme or u.netloc:
            continue
        local = unquote(u.path)
        dest = "index.html" if local in ("./","/") else local or name
        check((ROOT / dest).is_file(), f"{name}: missing target {raw}")
        if u.fragment and dest in pages:
            check(unquote(u.fragment) in pages[dest].ids, f"{name}: missing fragment {raw}")

url_to_page = {BASE + ("" if name == "index.html" else name): p for name,p in pages.items() if name != "404.html"}
for name,p in pages.items():
    if name == "404.html": continue
    check(p.language in ("en", "de"), f"{name}: missing or invalid lang")
    if p.language == "de":
        check(bool(p.alternates), f"{name}: missing language alternatives")
    if p.alternates:
        check(set(p.alternates) == {"en", "de", "x-default"}, f"{name}: unexpected hreflang set")
        check(p.alternates.get(p.language) in p.canonical, f"{name}: missing self-referencing hreflang")
        check(p.alternates.get("x-default") == p.alternates.get("en"), f"{name}: invalid x-default")
        for lang in ("en", "de"):
            target = url_to_page.get(p.alternates.get(lang))
            check(target is not None and target.language == lang and target.alternates == p.alternates, f"{name}: nonreciprocal or invalid {lang} hreflang")

ns = {"s":"http://www.sitemaps.org/schemas/sitemap/0.9"}
locs = [x.text for x in ET.parse(ROOT/"sitemap.xml").findall("s:url/s:loc", ns)]
expected = {BASE + ("" if p=="index.html" else p) for p in pages if p!="404.html"}
check(set(locs)==expected and len(locs)==len(expected), "sitemap mismatch")
check(bool(pages["index.html"].meta.get("google-site-verification")), "Google verification missing")
png=(ROOT/"social-preview.png").read_bytes()
check(png[:8]==b"\x89PNG\r\n\x1a\n" and int.from_bytes(png[16:20],"big")==1200 and int.from_bytes(png[20:24],"big")==630,"social image format/dimensions")
if "monitor.html" in pages or "monitor-de.html" in pages:
    for name,w,h in [("monitor-desktop.png",1440,1200),("monitor-mobile.png",390,1600)]:
        path=ROOT/name
        check(path.is_file(), f"{name}: screenshot missing")
        if path.is_file():
            png=path.read_bytes()
            check(png[:8]==b"\x89PNG\r\n\x1a\n" and int.from_bytes(png[16:20],"big")==w and int.from_bytes(png[20:24],"big")==h,f"{name}: screenshot format/dimensions")
if errors:
    raise SystemExit("\n".join(errors))
print(f"PASS: {len(expected)} indexable pages, local links, anchors, canonical URLs, sitemap, Google verification reciprocal hreflang and 1200x630 preview")
