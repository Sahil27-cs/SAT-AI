import urllib.request
import re

for base in ["https://sat-ai-sahil.vercel.app", "https://sat-ai-murex.vercel.app"]:
    print(f"\n=== Checking {base} ===")
    try:
        req = urllib.request.Request(base + "/", headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req).read().decode("utf-8")
        scripts = re.findall(r'/_next/static/[^"]+\.js', html)
        for s in scripts:
            if "app/page" in s or "255" in s or "4bd1b696" in s:
                content = urllib.request.urlopen(urllib.request.Request(base + s, headers={"User-Agent": "Mozilla/5.0"})).read().decode("utf-8", errors="ignore")
                matches = re.findall(r'https?://[a-zA-Z0-9\.\-_]+vercel\.app[^\s"\'\`]*', content)
                if matches:
                    print(f"  In {s}: URLs: {set(matches)}")
                if "/api/chat" in content or "/api/v1/chat" in content:
                    print(f"  In {s}: contains chat endpoint references")
    except Exception as e:
        print(f"  Error: {e}")
