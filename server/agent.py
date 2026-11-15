"""The parcel-finding agent.

One local model (MiniCPM-V 4.7 1B behind llama-server) does all the looking:
it reads the request, judges the photo, reads the shelf sign, reads the pickup
codes, points at the target label and re-reads it. What it reports decides
which step runs next. Number comparison (is shelf 8 the shelf in "8-2-5726"?)
is done in code, because that is where the small model was measured to slip.

Every step is emitted as an event so the app can show the run as it happens:

    think  - why the next step is being taken
    act    - a tool call (one or more model calls)
    see    - what the tool reported
    turn   - the plan changed because of what was seen
    ask    - the agent cannot go on and hands the question to the person
    done   - this photo is finished
"""
import base64
import io
import json
import re
import time
import urllib.error
import urllib.request

from PIL import Image, ImageDraw, ImageFont, ImageOps

# The model server is on this machine: never route these calls through a system proxy.
_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))

CODE_RE = re.compile(r"(\d{1,3})\s*-\s*(\d{1,2})\s*-\s*(\d{4})")
PHOTO_LONG_EDGE = 1280     # photos are normalised to the size the prompts were tested at
TILE_LONG_EDGE = 800

P_READ = "图中每个快递标签上都印着一个大号的四位数字。请读出所有能看清的四位数字，用空格分隔，只输出数字。"
P_QUALITY = ('这是一张快递货架的照片，我需要读出包裹标签上的四位数字。标签上的数字清晰可读吗？'
             '只输出 JSON：{"readable": true 或 false, "problem": "清晰" 或 "模糊" 或 "太暗" 或 "太远"}')
P_SIGN = ('照片里货架边缘的蓝色标牌上印着货架号和层号。这是几号货架？能看到哪几层？'
          '只输出 JSON：{"shelf": 数字, "levels": [数字]}。如果看不到标牌或读不出来，输出 {"shelf": null, "levels": []}')
P_REQUEST = ('从下面这段话里找出所有快递取件码。取件码可能是“货架号-层号-四位数字”，也可能只有四位或六位数字。'
             '把每个取件码按原文写出来，只输出 JSON 数组，没有就输出 []。不要编造原文里没有的数字。\n\n')
P_TRANSCRIBE = "请原样读出这张图片里的所有文字，不要解释，不要省略数字和符号。"
P_IMAGE_CODES = ('这张图片里有快递取件码吗？取件码可能是“货架号-层号-四位数字”，也可能只有四位或六位数字。'
                 '把每个取件码按图里写的样子写出来，只输出 JSON 数组，没有就输出 []。不要写验证码和电话号码。')
P_POINT = ('图里有多个贴着四位数字标签的快递。找到数字是 {code} 的那一个标签，给出它的位置，'
           '用 0 到 1000 的相对坐标，只输出 JSON：{{"box": [x1, y1, x2, y2]}}')


class Brain:
    """Client for the local model server. Counts every call."""

    def __init__(self, server, model="minicpm-v"):
        self.url = server.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.calls = 0
        self.seconds = 0.0

    def ask(self, prompt, image=None, max_tokens=200, long_edge=TILE_LONG_EDGE):
        content = []
        if image is not None:
            s = long_edge / max(image.size)
            if abs(s - 1) > 0.02:
                image = image.resize((max(1, round(image.width * s)), max(1, round(image.height * s))), Image.LANCZOS)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}})
        content.append({"type": "text", "text": prompt})
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "user", "content": content}]}
        data = json.dumps(body).encode()
        t0 = time.time()
        for attempt in range(3):
            req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
            try:
                with _LOCAL.open(req, timeout=120) as r:
                    text = json.load(r)["choices"][0]["message"].get("content") or ""
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                # a dropped local connection is not an answer; try again before giving up
                if attempt == 2:
                    raise
                time.sleep(0.5)
        self.seconds += time.time() - t0
        self.calls += 1
        return text.strip()


def _json(text):
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _codes(text):
    return [g[-4:] for g in re.findall(r"\d{4,}", text)]


def _near(a, b):
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


# ---------------------------------------------------------------- request

SHELF_RE = re.compile(r"(\d{1,3})\s*号?\s*货架\s*第?\s*(\d{1,2})\s*层\D{0,8}?(\d{4})(?!\d)")
KEYWORD_RE = re.compile(r"(?:取件码|取货码|提货码|取件号|取件编号)\s*[:：是为]?\s*([A-Za-z]?\d[\d\-]{2,10}\d)")
ALONE_RE = re.compile(r"(?<![\d:：.\-])(\d{4})(?![\d:：.\-])")
_ZH = {"零": "0", "〇": "0", "一": "1", "幺": "1", "二": "2", "两": "2", "三": "3", "四": "4",
       "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
_ZH_RUN = re.compile(r"[零〇一幺二两三四五六七八九十]+")


def spoken_to_written(text):
    """Turn the way a code is said aloud, or mis-punctuated by speech recognition, into the written form.

    八杠二杠五七二六 / 8 杠 2 杠 5726 / 8-2,1032 / 6′1′1402  ->  8-2-5726 / 8-2-1032 / 6-1-1402
    十二号货架第三层 -> 12号货架第3层.  Words such as 一下 or 一个 are left alone.
    Only used for finding codes, never shown back to the person.
    """
    def codeish(ch):
        return ch != "" and (ch in "杠号层横-第" or ch.isdigit())

    def convert(m):
        run = m.group(0)
        before = text[m.start() - 1] if m.start() > 0 else ""
        after = text[m.end()] if m.end() < len(text) else ""
        # a lone numeral is part of a word (一下, 一个, 三个) unless it sits next to something code-like
        if len(run) == 1 and run != "十" and not codeish(before) and not codeish(after):
            return run
        if "十" in run:
            a, _, b = run.partition("十")
            if len(a) > 1 or "十" in b:
                return "".join(_ZH.get(c, c) for c in run)      # not a simple 十-number; leave as digits where possible
            return str((int(_ZH[a]) if a else 1) * 10 + (int(_ZH[b[0]]) if b else 0)) + "".join(_ZH[c] for c in b[1:])
        return "".join(_ZH[c] for c in run)
    t = _ZH_RUN.sub(convert, text)
    t = re.sub(r"\s*(?:横杠|杠|横|减|－|—|–)\s*", "-", t)
    # speech recognition sometimes writes the hyphen as a dot or a prime: 6′1′1402, 8‧2‧5726
    t = re.sub(r"(?<=\d)\s*[‧·•′’'．]\s*(?=\d)", "-", t)
    # ... or breaks the code before its last four digits: 8-2,1032
    return re.sub(r"(?<![\d\-])(\d{1,3}-\d{1,2})\s*[,，、\s]\s*(\d{4})(?!\d)", r"\1-\2", t)


def make_parcel(shelf, level, tail):
    """shelf and level are 0 when the code does not say; the person finds the shelf themselves."""
    code = f"{shelf}-{level}-{tail}" if shelf and level else (f"{shelf}-{tail}" if shelf else tail)
    return {"code": code, "shelf": shelf, "level": level, "tail": tail, "status": "pending", "note": ""}


def parse_request(brain, text):
    """Pickup codes in whatever the person typed, pasted or said.

    Accepted: 8-2-5726, 8号货架2层的5726, 取件码 12-6029, 取件码 5726, and in a short message of the
    person's own a bare 5726. A pasted notice (long, or with 【】) only yields codes it labels as such,
    so phone numbers, verification codes and opening hours stay out. The model adds codes phrased in
    other ways, but only ones whose last four digits really occur in the text.
    """
    plain = spoken_to_written(text)
    found = {}                                   # tail -> parcel; a fuller form wins over a bare tail

    def add(shelf, level, tail):
        known = found.get(tail)
        if known is None or (shelf and not known["shelf"]) or (level and not known["level"]):
            found[tail] = make_parcel(shelf, level, tail)

    for a, b, c in CODE_RE.findall(plain):
        add(int(a), int(b), c)
    for a, b, c in SHELF_RE.findall(plain):
        add(int(a), int(b), c)
    for token in KEYWORD_RE.findall(plain):
        parts = re.findall(r"\d+", token)
        tail = parts[-1][-4:]
        if len(parts) >= 3:
            add(int(parts[-3]), int(parts[-2]), tail)
        elif len(parts) == 2:
            add(int(parts[0]), 0, tail)
        else:
            add(0, 0, tail)
    own_words = len(plain) <= 40 and "【" not in plain
    if own_words:
        for tail in ALONE_RE.findall(plain):
            add(0, 0, tail)

    source = "pattern"
    answer = _json(brain.ask(P_REQUEST + plain, max_tokens=160))
    if isinstance(answer, list):
        for item in answer:
            raw = str(item.get("code", "") if isinstance(item, dict) else item)
            parts = re.findall(r"\d+", spoken_to_written(raw))
            if not parts or len(parts[-1]) < 4:
                continue
            tail = parts[-1][-4:]
            # the number must stand in the text as a whole (never invent a parcel), and must not be
            # a verification code, phone number or the like
            at = re.search(r"(?<!\d)" + parts[-1] + r"(?!\d)", plain)
            if at is None or re.search(r"验证码|校验码|密码|电话|手机|尾号|联系", plain[max(0, at.start() - 8):at.start()]):
                continue
            if tail in found:
                source = "model"
                continue
            if len(parts) >= 3:
                add(int(parts[-3]), int(parts[-2]), tail)
            elif len(parts) == 2:
                add(int(parts[0]), 0, tail)
            else:
                add(0, 0, tail)
            source = "model"
    return list(found.values()), source


def parse_request_image(brain, image):
    """Pickup codes in a picture of the pickup message: an SMS screenshot, a photo of the phone, a pickup slip.

    The model first reads out everything written in the picture; the codes are then taken from that
    text the same way as from a typed message, so a code is only accepted if it was actually read.
    If nothing turns up, the model is asked once more for the codes alone.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    long_edge = min(1600, max(image.size))
    text = brain.ask(P_TRANSCRIBE, image, max_tokens=700, long_edge=long_edge)
    parcels, source = parse_request(brain, text)
    if not parcels:
        again = brain.ask(P_IMAGE_CODES, image, max_tokens=120, long_edge=long_edge)
        parcels, source = parse_request(brain, again)
        text = text + " " + again
    return parcels, text, source


def make_plan(parcels):
    groups = {}
    for p in parcels:
        if p["status"] != "found":
            groups.setdefault((p["shelf"], p["level"]), []).append(p["tail"])
    order = sorted(groups, key=lambda k: (k[0] == 0, k))
    return [{"shelf": s, "level": l, "tails": groups[(s, l)]} for s, l in order]


def normalise(photo):
    # phone cameras store the sensor image plus a "rotate me" tag; apply it, or shelves arrive sideways
    photo = ImageOps.exif_transpose(photo).convert("RGB")
    s = PHOTO_LONG_EDGE / max(photo.size)
    if s < 1:
        photo = photo.resize((round(photo.width * s), round(photo.height * s)), Image.LANCZOS)
    return photo


def tile_boxes(W, H, overlap=0.2):
    cols, rows = max(2, round(W / 320)), max(2, round(H / 320))
    tw, th = W / cols, H / rows
    return [(int(max(0, c * tw - tw * overlap)), int(max(0, r * th - th * overlap)),
             int(min(W, (c + 1) * tw + tw * overlap)), int(min(H, (r + 1) * th + th * overlap)))
            for r in range(rows) for c in range(cols)]


def check_quality(brain, photo):
    j = _json(brain.ask(P_QUALITY, photo, max_tokens=60, long_edge=PHOTO_LONG_EDGE))
    if not isinstance(j, dict) or not isinstance(j.get("readable"), bool):
        return {"readable": True, "problem": "未知"}      # an unparseable answer must not block the person
    return {"readable": j["readable"], "problem": str(j.get("problem") or "")}


def read_sign(brain, photo):
    j = _json(brain.ask(P_SIGN, photo, max_tokens=60, long_edge=PHOTO_LONG_EDGE))
    if not isinstance(j, dict):
        return {"shelf": None, "levels": []}
    levels = [x for x in (_int(v) for v in (j.get("levels") or [])) if x is not None]
    return {"shelf": _int(j.get("shelf")), "levels": levels}


def point_at(brain, photo, tile, code, long_edge=TILE_LONG_EDGE):
    """Ask where the label is inside one tile; returns a box in photo pixels or None.

    On a single tile the answer was right 10 times out of 10 when checked by re-reading.
    On the whole photo it lands about one parcel off, which is still a good place to start looking.
    """
    crop = photo.crop(tile)
    m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
                  brain.ask(P_POINT.format(code=code), crop, max_tokens=60, long_edge=long_edge))
    if not m:
        return None
    x1, y1, x2, y2 = (int(v) for v in m.groups())
    if x2 <= x1 or y2 <= y1:
        return None
    w, h = crop.size
    return (tile[0] + x1 * w // 1000, tile[1] + y1 * h // 1000, tile[0] + x2 * w // 1000, tile[1] + y2 * h // 1000)


def padded(box, W, H, frac=0.34, least=14):
    px, py = max(least, int((box[2] - box[0]) * frac)), max(least, int((box[3] - box[1]) * frac))
    return (max(0, box[0] - px), max(0, box[1] - py), min(W, box[2] + px), min(H, box[3] + py))


def narrow_by_grid(brain, photo, region, code, min_size=110):
    """Fallback locator: keep splitting 3x3 and keeping the pieces that still show the code."""
    while max(region[2] - region[0], region[3] - region[1]) > min_size:
        x0, y0, x1, y1 = region
        tw, th = (x1 - x0) / 3, (y1 - y0) / 3
        pieces = [(int(max(x0, x0 + c * tw - tw * .2)), int(max(y0, y0 + r * th - th * .2)),
                   int(min(x1, x0 + (c + 1) * tw + tw * .2)), int(min(y1, y0 + (r + 1) * th + th * .2)))
                  for r in range(3) for c in range(3)]
        hits = [p for p in pieces if code in _codes(brain.ask(P_READ, photo.crop(p)))]
        if not hits:
            break
        ix = (max(b[0] for b in hits), max(b[1] for b in hits), min(b[2] for b in hits), min(b[3] for b in hits))
        region = ix if ix[2] - ix[0] > 20 and ix[3] - ix[1] > 20 else hits[0]
    return region


def annotate(photo, marks, path):
    """Circle each mark. Sizes scale with the photo so the ring still reads on a phone-width view."""
    out = photo.copy()
    d = ImageDraw.Draw(out)
    unit = max(out.size) / 1280
    line, pad, size = round(11 * unit), round(18 * unit), round(52 * unit)
    try:
        font = ImageFont.truetype("arialbd.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    for m in marks:
        x0, y0, x1, y1 = m["box"]
        color = (239, 43, 45) if m["verified"] else (255, 170, 0)
        ring = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        d.ellipse((ring[0] - 3, ring[1] - 3, ring[2] + 3, ring[3] + 3), outline=(255, 255, 255), width=line + 6)
        d.ellipse(ring, outline=color, width=line)
        label = m["tail"] + ("" if m["verified"] else " ?")
        tw, th = d.textbbox((0, 0), label, font=font)[2:]
        tx = min(max(0, (x0 + x1) // 2 - tw // 2), out.width - tw - 12)
        ty = ring[1] - th - round(30 * unit)
        if ty < 0:
            ty = ring[3] + round(10 * unit)
        d.rounded_rectangle((tx - 12, ty - 4, tx + tw + 12, ty + th + 16), radius=round(10 * unit), fill=color)
        d.text((tx, ty), label, fill=(255, 255, 255), font=font)
    out.save(path, quality=92)


# ---------------------------------------------------------------- the loop

def place(shelf, level):
    if not shelf:
        return "货架未知"
    if not level:
        return f"{shelf} 号货架"
    return f"{shelf} 号货架第 {level} 层"


def run_photo(brain, task, photo, emit, annotated_path, force=False):
    """Work through one photo for a task. Mutates task['parcels']; returns a summary dict.

    emit(kind, text, **extra) is called for every step.
    """
    calls0, turns, asks = brain.calls, 0, 0
    W, H = photo.size
    rel = lambda b: [round(b[0] / W, 4), round(b[1] / H, 4), round(b[2] / W, 4), round(b[3] / H, 4)]
    pending = [p for p in task["parcels"] if p["status"] != "found"]
    tails = lambda ps: "、".join(p["tail"] for p in ps)

    def finish(status, message, marks=()):
        annotate(photo, list(marks), annotated_path)
        summary = {"status": status, "message": message, "calls": brain.calls - calls0, "turns": turns, "asks": asks,
                   "found": [m["tail"] for m in marks], "size": [W, H],
                   "marks": [{"tail": m["tail"], "box": rel(m["box"]), "verified": m["verified"]} for m in marks]}
        emit("done", message, summary=summary)
        return summary

    # 1. is the photo usable at all?
    emit("think", "先看这张照片能不能用，看不清就不往下找。")
    emit("act", "检查照片质量()")
    q = check_quality(brain, photo)
    if not q["readable"]:
        emit("see", f"照片{q['problem'] or '看不清'}，标签上的数字读不出来。")
        asks += 1
        emit("ask", f"这张照片{q['problem'] or '看不清'}，我不猜。请靠近一点或换个角度重拍。", need="retake")
        return finish("retake", "这张照片看不清，请重拍一张。")
    emit("see", "标签上的数字清晰可读。")

    # 2. which shelf is this? The person chose where to stand, so the sign only decides the order:
    #    parcels whose code names this shelf are looked for first, the rest after.
    emit("think", "先看这是哪个货架，取件码对得上的先找。")
    emit("act", "读货架标牌()")
    sign = read_sign(brain, photo)
    if sign["shelf"] is None:
        emit("see", "没有读到货架标牌，那就不分先后，照片里全部找一遍。")
        here, mismatched = pending, []
    else:
        lv = "、".join(str(x) for x in sign["levels"])
        emit("see", f"这是 {sign['shelf']} 号货架" + (f"，第 {lv} 层。" if lv else "。"), sign=sign)
        def says_other_shelf(p):
            if not p["shelf"]:
                return False
            return p["shelf"] != sign["shelf"] or bool(p["level"] and sign["levels"] and p["level"] not in sign["levels"])
        mismatched = [p for p in pending if says_other_shelf(p)]
        here = [p for p in pending if not says_other_shelf(p)] + mismatched
        if mismatched and not force:
            emit("think", f"{tails(mismatched)} 的取件码写的不是这一面，但你可能更清楚在哪，我还是在这张里找一遍，放在最后找。")
        else:
            emit("think", f"在这张里找 {tails(here)}。")
    away = []

    # 3 + 4. for each wanted parcel: guess where it is, read tiles outward from the guess and
    # stop at the first one that shows the code, then pin it down and confirm. Tiles already
    # read are remembered, so a second parcel never pays for the same tile twice.
    boxes = tile_boxes(W, H)
    unread = list(range(len(boxes)))
    seen = {}

    def read_tile(i):
        unread.remove(i)
        for c in _codes(brain.ask(P_READ, photo.crop(boxes[i]))):
            seen.setdefault(c, []).append(boxes[i])
        emit("progress", "", tile=rel(boxes[i]), index=len(boxes) - len(unread), total=len(boxes))

    marks = []
    for p in here:
        code = p["tail"]
        if code in seen:
            emit("think", f"{code} 刚才读别的块时已经见过，不用再扫。")
        elif unread:
            emit("think", f"不必整面读一遍：先估一下 {code} 大概在哪，从那附近读起，读到就停。")
            emit("act", f"估计位置({code})")
            guess = point_at(brain, photo, (0, 0, W, H), code, long_edge=PHOTO_LONG_EDGE)
            if guess:
                gx, gy = (guess[0] + guess[2]) / 2, (guess[1] + guess[3]) / 2
                order = sorted(unread, key=lambda i: ((boxes[i][0] + boxes[i][2]) / 2 - gx) ** 2 + ((boxes[i][1] + boxes[i][3]) / 2 - gy) ** 2)
                emit("see", "有个大概位置，不一定准，从它所在的那一块读起。", box=rel(padded(guess, W, H)))
            else:
                order = list(unread)
                emit("see", "估不出位置，按顺序读。")
            emit("act", f"分块扫描(最多 {len(order)} 块)")
            before = len(unread)
            for i in order:
                read_tile(i)
                if code in seen:
                    break
            used = before - len(unread)
            if code in seen:
                emit("see", f"读了 {used} 块就读到 {code}" + (f"，剩下 {len(unread)} 块不用看了。" if unread else "。"))
            else:
                emit("see", f"{used} 块全部读完，没有 {code}。")
        if code not in seen:
            similar = sorted(c for c in seen if _near(c, code))
            confirmed = False
            for s in similar:
                turns += 1
                emit("turn", f"没读到 {code}，但有只差一位的 {s}，可能是读错了。放大那个位置再读一遍。")
                emit("act", f"指出位置({s})")
                box = point_at(brain, photo, seen[s][0], s)
                if box is None:
                    continue
                region = padded(box, W, H)
                again = _codes(brain.ask(P_READ, photo.crop(region)))
                if code in again:
                    emit("see", f"放大后读到的是 {code}，刚才那次是误读。", box=rel(region))
                    marks.append({"tail": code, "box": region, "verified": True})
                    p["status"], p["note"] = "found", "放大后确认"
                    confirmed = True
                    break
                emit("see", f"放大后仍是 {'、'.join(again) or '读不出'}，不是 {code}。", box=rel(region))
            if not confirmed:
                p["note"] = "这张照片里没有找到" + (f"，有相近的 {'、'.join(similar)}，请核对取件码" if similar else "")
                emit("see", f"{code} 在这张照片里没有找到。" + ("可能被遮住，或者不在这一层。" if not similar else ""))
            continue

        emit("act", f"指出位置({code})", tile=rel(seen[code][0]))
        box = point_at(brain, photo, seen[code][0], code)
        region = padded(box, W, H) if box else None
        emit("act", "复核()", box=rel(region) if region else None)
        again = _codes(brain.ask(P_READ, photo.crop(region))) if region else []
        if code in again and len(again) <= 2:
            emit("see", f"把指的位置单独再读一遍，仍是 {code}，两次一致。", box=rel(region))
            marks.append({"tail": code, "box": region, "verified": True})
            p["status"], p["note"] = "found", "两次读数一致"
            continue

        turns += 1
        emit("turn", (f"指的位置读出来是 {'、'.join(again)}，对不上。" if again else "没能指出准确位置。") + "改用逐块缩小的办法重新定位。")
        emit("act", "逐块缩小()", tile=rel(seen[code][0]))
        region = padded(narrow_by_grid(brain, photo, seen[code][0], code), W, H, frac=0.1, least=12)
        emit("act", "复核()", box=rel(region))
        again = _codes(brain.ask(P_READ, photo.crop(region)))
        ok = code in again
        emit("see", f"缩小后再读，{'仍是 ' + code + '，确认' if ok else '读到 ' + ('、'.join(again) or '空') + '，没能确认'}。", box=rel(region))
        marks.append({"tail": code, "box": region, "verified": ok})
        p["status"], p["note"] = ("found", "缩小范围后确认") if ok else ("pending", "位置不确定，已用橙色圈出，请自己核对")

    # 5. say what happened and what is still owed
    found = [p for p in here if p["status"] == "found"]
    missing = [p for p in here if p["status"] != "found"]
    left = [p for p in task["parcels"] if p["status"] != "found"]
    parts = []
    if found:
        parts.append(f"找到 {tails(found)} 了，已经在照片上圈出来，取件时再核对一下姓名。")
    for p in missing:
        if p in mismatched:
            parts.append(f"{p['tail']} 没在这张里找到，取件码上写的是 {place(p['shelf'], p['level'])}，可以去那边拍一张。")
        else:
            parts.append(f"{p['tail']} 没在这张里找到，可能被挡住了，换个角度或换一排货架再拍一张。")
    if not left:
        parts.append("全部找齐了。")
    return finish("all_found" if not left else "more", "".join(parts), marks)
