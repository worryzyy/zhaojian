"""How many pickup codes does a model read from a shelf photo, tile by tile?

    python read_compare.py <photo> <server> <model-alias> "<prompt>" [cols rows]

Prints the codes read per tile, then recall against the codes a person
counted on samples/shelf.jpg (TRUTH below). Extra codes are listed too,
because a misread code is worse than a missed one.
"""
import base64
import io
import json
import re
import sys
import time
import urllib.request

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

# Codes fully visible in samples/shelf.jpg, counted by eye from the photo.
TRUTH = """3262 2299 2805 9434 0894 8932 5017 7626 5965
0030 4009 1365 5989 3897 4520 0202 8354 5358 5341
3388 6489 9650 2042 6269 3951 3387 9586 4435
0843 5726 8203 3128 6209 9107 9442
3700 9745 1032 1470 1852 4085 4339 4470 0348 0166 8631
9425 4457 6103 3259 0045 1402 0498""".split()

photo, server, model, prompt = sys.argv[1:5]
cols, rows = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (4, 3)
img = Image.open(photo).convert("RGB")
W, H = img.size
url = server.rstrip("/") + "/v1/chat/completions"

seen, total = {}, 0.0
tw, th = W / cols, H / rows
for r in range(rows):
    for c in range(cols):
        box = (int(max(0, c * tw - tw * .15)), int(max(0, r * th - th * .15)),
               int(min(W, (c + 1) * tw + tw * .15)), int(min(H, (r + 1) * th + th * .15)))
        tile = img.crop(box)
        s = 800 / max(tile.size)
        tile = tile.resize((round(tile.width * s), round(tile.height * s)), Image.LANCZOS)
        buf = io.BytesIO()
        tile.save(buf, format="PNG")
        body = {"model": model, "max_tokens": 256, "temperature": 0, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
            {"type": "text", "text": prompt}]}]}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as resp:
            text = json.load(resp)["choices"][0]["message"]["content"]
        dt = time.time() - t0
        total += dt
        codes = [g[-4:] for g in re.findall(r"\d{4,}", text)]
        for code in codes:
            seen.setdefault(code, 0)
            seen[code] += 1
        print(f"[r{r} c{c}] {dt:5.2f}s  {' '.join(codes)}")

hit = [c for c in TRUTH if c in seen]
miss = [c for c in TRUTH if c not in seen]
extra = sorted(c for c in seen if c not in TRUTH)
print(f"\ntime: {total:.1f}s for {cols * rows} tiles ({total / (cols * rows):.2f}s each)")
print(f"read correctly: {len(hit)}/{len(TRUTH)}")
print(f"missed: {' '.join(miss) or '-'}")
print(f"not in the hand count (misreads or partly hidden labels): {' '.join(extra) or '-'}")
