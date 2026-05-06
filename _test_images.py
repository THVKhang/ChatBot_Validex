import sys, json, requests, re
sys.stdout.reconfigure(encoding='utf-8')
r = requests.post('http://localhost:8000/api/chat/stream', json={'prompt': 'children policy in australia'})
lines = [json.loads(x.replace('data: ','')) for x in r.text.strip().split('\n') if x.startswith('data:')]
f = [l for l in lines if 'generated' in l][-1]
draft = f['generated']['draft']
title = f['generated']['title']
images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', draft)
print("Title:", title)
print("Images found:", len(images))
for i, (alt, url) in enumerate(images):
    print(f"  [{i+1}] alt={alt[:60]}")
    print(f"       url={url[:100]}")
