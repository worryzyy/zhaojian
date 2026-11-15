"""Find a parcel by its pickup code in a shelf photo and circle it.

The recognition model is MiniCPM-OCR 1.0 served locally by llama-server
(OpenAI-compatible endpoint). Nothing leaves this machine.

    python find_parcel.py <photo> <code> [<code> ...] [--server URL] [--out FILE]

Search strategy, driven entirely by what the local model reads:
  1. scan   - cut the photo into overlapping tiles, read each, stop at the
              first tile whose text contains the code
  2. narrow - cut that tile into smaller overlapping pieces and keep the ones
              that still contain the code; repeat until the region is about
              one label in size
  3. verify - read the final region once more on its own; only a confirmed
              hit is circled
  4. report - if nothing matches, list the closest codes that were seen
"""
import argparse
import base64
import io
import json
import re
import sys
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

PROMPT = "Text Recognition:"
TARGET_LONG_EDGE = 800   # tiles are resized so the model sees enough pixels
MIN_REGION = 110         # stop narrowing below this size (source pixels)
OVERLAP = 0.2


class Reader:
    """Thin client for the local MiniCPM-OCR server; counts calls and time."""

    def __init__(self, server, model="minicpm-ocr", prompt=PROMPT):
        self.url = server.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.prompt = prompt
        self.calls = 0
        self.seconds = 0.0

    def read(self, image):
        long_edge = max(image.size)
        if long_edge != TARGET_LONG_EDGE:
            s = TARGET_LONG_EDGE / long_edge
            image = image.resize((max(1, round(image.width * s)), max(1, round(image.height * s))),
                                 Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        body = {
            "model": self.model,
            "max_tokens": 256,
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
                {"type": "text", "text": self.prompt},
            ]}],
        }
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            text = json.load(r)["choices"][0]["message"]["content"]
        self.seconds += time.time() - t0
        self.calls += 1
        return text


def digit_groups(text):
    return re.findall(r"\d{3,}", text)


def normalize(code):
    """'8-2-5726' and '5726' both mean the 4-digit tail printed large on the label."""
    return re.findall(r"\d+", code)[-1]


def contains(text, code):
    return any(g == code or g.endswith(code) for g in digit_groups(text))


def tiles(box, cols, rows):
    x0, y0, x1, y1 = box
    tw, th = (x1 - x0) / cols, (y1 - y0) / rows
    ox, oy = tw * OVERLAP, th * OVERLAP
    for r in range(rows):
        for c in range(cols):
            yield (int(max(x0, x0 + c * tw - ox)), int(max(y0, y0 + r * th - oy)),
                   int(min(x1, x0 + (c + 1) * tw + ox)), int(min(y1, y0 + (r + 1) * th + oy)))


def intersect(boxes):
    x0 = max(b[0] for b in boxes); y0 = max(b[1] for b in boxes)
    x1 = min(b[2] for b in boxes); y1 = min(b[3] for b in boxes)
    return (x0, y0, x1, y1) if x1 - x0 > 20 and y1 - y0 > 20 else None


def distance(a, b):
    if len(a) != len(b):
        return 9
    return sum(1 for x, y in zip(a, b) if x != y)


def find(img, code, reader, log):
    W, H = img.size
    seen = set()

    # 1. scan
    cols = max(2, round(W / 330)); rows = max(2, round(H / 330))
    region = None
    for box in tiles((0, 0, W, H), cols, rows):
        text = reader.read(img.crop(box))
        seen.update(digit_groups(text))
        if contains(text, code):
            region = box
            log(f"scan: code seen in tile {box}")
            break
    if region is None:
        near = sorted(g for g in seen if distance(g[-len(code):], code) == 1)
        log(f"scan: not found in {cols * rows} tiles; {len(seen)} codes read")
        return {"code": code, "found": False, "similar": near, "codes_seen": len(seen)}

    # 2. narrow
    while max(region[2] - region[0], region[3] - region[1]) > MIN_REGION:
        hits = [b for b in tiles(region, 3, 3) if contains(reader.read(img.crop(b)), code)]
        if not hits:
            log(f"narrow: no smaller piece still shows the code, keeping {region}")
            break
        region = intersect(hits) or hits[0]
        log(f"narrow: {len(hits)} piece(s) -> {region}")

    # 3. verify
    pad = 12
    check = (max(0, region[0] - pad), max(0, region[1] - pad),
             min(W, region[2] + pad), min(H, region[3] + pad))
    text = reader.read(img.crop(check))
    verified = contains(text, code)
    log(f"verify: region {check} reads {' '.join(text.split())[:60]!r} -> {'confirmed' if verified else 'NOT confirmed'}")
    return {"code": code, "found": True, "verified": verified, "box": check}


def annotate(img, results, out_path):
    out = img.copy()
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arialbd.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    for r in results:
        if not r.get("found"):
            continue
        x0, y0, x1, y1 = r["box"]
        color = (255, 30, 30) if r["verified"] else (255, 170, 0)
        m = 14
        for w in range(6):
            d.ellipse((x0 - m - w, y0 - m - w, x1 + m + w, y1 + m + w), outline=color)
        label = r["code"] + ("" if r["verified"] else " ?")
        tx, ty = x0 - m, max(0, y0 - m - 40)
        d.rectangle(d.textbbox((tx, ty), label, font=font), fill=color)
        d.text((tx, ty), label, fill=(255, 255, 255), font=font)
    out.save(out_path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("codes", nargs="+")
    ap.add_argument("--server", default="http://127.0.0.1:18499")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="minicpm-ocr", help="model alias served at --server")
    ap.add_argument("--prompt", default=PROMPT, help="reading prompt; the OCR model uses 'Text Recognition:'")
    a = ap.parse_args()

    img = Image.open(a.photo).convert("RGB")
    reader = Reader(a.server, a.model, a.prompt)
    results = []
    for raw in a.codes:
        code = normalize(raw)
        print(f"== {raw} (looking for {code})")
        c0, s0 = reader.calls, reader.seconds
        r = find(img, code, reader, lambda m: print("   " + m))
        r["calls"] = reader.calls - c0
        r["seconds"] = round(reader.seconds - s0, 1)
        results.append(r)
        if r["found"]:
            state = "FOUND" if r["verified"] else "FOUND (unconfirmed)"
            print(f"   {state} at {r['box']}  [{r['calls']} reads, {r['seconds']}s]")
        else:
            hint = f"; similar codes seen: {', '.join(r['similar'])}" if r["similar"] else ""
            print(f"   NOT FOUND - try another shelf or retake the photo{hint}  "
                  f"[{r['calls']} reads, {r['seconds']}s]")

    out_path = a.out or re.sub(r"\.\w+$", "", a.photo) + "_found.jpg"
    annotate(img, results, out_path)
    print(f"annotated photo: {out_path}")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
