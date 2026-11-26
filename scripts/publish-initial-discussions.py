"""Publish the curated initial discussions, skipping exact existing titles."""
import argparse, json, os, pathlib, subprocess

parser=argparse.ArgumentParser()
parser.add_argument("--apply",action="store_true")
args=parser.parse_args()
posts=json.loads((pathlib.Path(__file__).resolve().parents[1]/"docs/maintainer/discussion-posts.json").read_text(encoding="utf-8"))
def graphql(query,variables):
    p=subprocess.run(["gh","api","graphql","--input","-"],input=json.dumps({"query":query,"variables":variables}),text=True,encoding="utf-8",capture_output=True,check=True)
    response=json.loads(p.stdout)
    if response.get("errors"):raise RuntimeError(response["errors"])
    return response["data"]
query="""query($cursor:String){repository(owner:"spk-c",name:"ninfer-all-rtx4090"){id hasDiscussionsEnabled discussionCategories(first:50){nodes{id name}} discussions(first:100,after:$cursor){nodes{id title url} pageInfo{hasNextPage endCursor}}}}"""
cursor=None
existing={}
while True:
    repo=graphql(query,{"cursor":cursor})["repository"]
    if not repo["hasDiscussionsEnabled"]:raise SystemExit("Discussions are disabled")
    page=repo["discussions"]
    existing.update({n["title"]:n["url"] for n in page["nodes"]})
    if not page["pageInfo"]["hasNextPage"]:break
    cursor=page["pageInfo"]["endCursor"]
categories={c["name"].casefold():c["id"] for c in repo["discussionCategories"]["nodes"]}
mutation="""mutation($input:CreateDiscussionInput!){createDiscussion(input:$input){discussion{id title url}}}"""
for post in posts:
    title=post["title"]
    if title in existing:
        print("EXISTS",title,existing[title],flush=True);continue
    category=next((x for x in post["categories"] if x.casefold() in categories),None)
    if not category:raise SystemExit("No matching category for "+title)
    if not args.apply:
        print("PREVIEW",category,title,flush=True);continue
    d=graphql(mutation,{"input":{"repositoryId":repo["id"],"categoryId":categories[category.casefold()],"title":title,"body":post["body"]}})["createDiscussion"]["discussion"]
    print("PUBLISHED",d["title"],d["url"],flush=True)
