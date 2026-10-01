import urllib.request
import re

url = "https://sat-ai-murex.vercel.app/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
html = urllib.request.urlopen(req).read().decode("utf-8")
scripts = re.findall(r'/_next/static/[^"]+\.js', html)
print("Found scripts:", len(scripts))
for s in scripts:
    s_url = "https://sat-ai-murex.vercel.app" + s
    content = urllib.request.urlopen(urllib.request.Request(s_url, headers={"User-Agent": "Mozilla/5.0"})).read().decode("utf-8", errors="ignore")
    # Search for api endpoints or backend urls
    matches = re.findall(r'https?://[a-zA-Z0-9\.\-_]+vercel\.app[^\s"\'\`]*', content)
    chat_calls = [line for line in content.split(";") if "/api/chat" in line or "/api/v1/chat" in line]
    if matches or chat_calls:
        print(f"\nIn {s}:")
        if matches:
            print("  Backend URLs found:", set(matches))
        if chat_calls:
            print("  Chat code snippet:", chat_calls[:2])
