import urllib.request
import re

url = "https://sat-ai-sahil.vercel.app/_next/static/chunks/app/page-81215066390cb52f.js"
content = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})).read().decode("utf-8")

# Find askAgent in compiled js
idx = content.find("askAgent")
if idx != -1:
    print("Found askAgent at", idx)
    print(content[max(0, idx-100):min(len(content), idx+1500)])
else:
    print("askAgent not found by name, searching for fetch(/api/chat) or similar:")
    matches = re.findall(r'.{0,100}/api/chat.{0,200}', content)
    for m in matches:
        print("-->", m)
