"""Can a small local model make the agent's decisions?

Runs five kinds of judgement against a local OpenAI-compatible server
(llama-server with MiniCPM-V 4.7 1B) and prints each raw answer next to
the expected one, so a wrong or malformed answer is visible, not hidden.

    python brain_eval.py <shelf-photo> [--server http://127.0.0.1:18500]
"""
import argparse
import base64
import io
import json
import re
import sys
import time
import urllib.request

from PIL import Image, ImageEnhance, ImageFilter

sys.stdout.reconfigure(encoding="utf-8")


class Model:
    def __init__(self, server, name):
        self.url = server.rstrip("/") + "/v1/chat/completions"
        self.name = name

    def ask(self, prompt, image=None, max_tokens=200, long_edge=1280):
        content = []
        if image is not None:
            if max(image.size) > long_edge:
                s = long_edge / max(image.size)
                image = image.resize((round(image.width * s), round(image.height * s)), Image.LANCZOS)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=92)
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}})
        content.append({"type": "text", "text": prompt})
        body = {"model": self.name, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "user", "content": content}]}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.load(r)
        msg = out["choices"][0]["message"]
        return (msg.get("content") or "").strip(), time.time() - t0


def parse_json(text):
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def as_int(v):
    """The model sometimes writes numbers as strings ("8"); the value is what matters."""
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def sign_ok(j):
    return isinstance(j, dict) and as_int(j.get("shelf")) == 8 and 2 in [as_int(x) for x in (j.get("levels") or [])]


results = []


def record(group, name, ok, answer, expected, seconds):
    results.append((group, ok))
    flat = " ".join(answer.split())
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({seconds:.1f}s)")
    print(f"         expected: {expected}")
    print(f"         answer:   {flat[:260]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--server", default="http://127.0.0.1:18500")
    ap.add_argument("--model", default="minicpm-v")
    a = ap.parse_args()
    m = Model(a.server, a.model)
    photo = Image.open(a.photo).convert("RGB")
    W, H = photo.size

    # ---- 1. read the shelf sign -------------------------------------------------
    print("\n1. 读货架标牌")
    q = ('照片里货架边缘的蓝色标牌上印着货架号和层号。这是几号货架？能看到哪几层？'
         '只输出 JSON：{"shelf": 数字, "levels": [数字]}')
    ans, dt = m.ask(q, photo)
    j = parse_json(ans)
    record("sign", "整张照片", sign_ok(j), ans, 'shelf=8, levels 含 2', dt)
    strip = photo.crop((int(W * 0.05), int(H * 0.50), int(W * 0.50), int(H * 0.66)))
    ans, dt = m.ask(q, strip)
    j = parse_json(ans)
    record("sign", "只看标牌附近的裁剪", sign_ok(j), ans, 'shelf=8, levels 含 2', dt)

    # ---- 2. is the photo usable? ------------------------------------------------
    print("\n2. 判断照片能不能用")
    q = ('这是一张快递货架的照片，我需要读出包裹标签上的四位数字。标签上的数字清晰可读吗？'
         '只输出 JSON：{"readable": true 或 false, "problem": "清晰" 或 "模糊" 或 "太暗" 或 "太远"}')
    variants = [
        ("原图", photo, True),
        ("严重模糊", photo.filter(ImageFilter.GaussianBlur(9)), False),
        ("很暗", ImageEnhance.Brightness(photo).enhance(0.12), False),
        ("轻微模糊", photo.filter(ImageFilter.GaussianBlur(1.2)), True),
    ]
    for name, im, want in variants:
        ans, dt = m.ask(q, im)
        j = parse_json(ans)
        record("quality", name, isinstance(j, dict) and j.get("readable") is want, ans, f"readable={str(want).lower()}", dt)

    # ---- 3. understand the request ----------------------------------------------
    print("\n3. 理解取件请求（纯文字）")
    q = ('从下面这段话里找出所有快递取件码。取件码的格式是“货架号-层号-四位数字”。'
         '只输出 JSON 数组，每项为 {"code": "原样的取件码", "shelf": 数字, "level": 数字}。没有取件码就输出 []。\n\n')
    cases = [
        ("标准短信", "【菜鸟驿站】您的包裹已到站，取件码 8-2-5726、6-1-0498，请凭码领取。", ["8-2-5726", "6-1-0498"]),
        ("口语", "帮我找一下 8-2-9442 这个快递，还有一个是6-1-1402", ["8-2-9442", "6-1-1402"]),
        ("带干扰数字", "【驿站】尾号 3310 的用户，您有 1 个包裹，取件码 12-3-0045，营业至 21:00，电话 13800001234。", ["12-3-0045"]),
        ("没有取件码", "你好，请问驿站几点关门？", []),
    ]
    for name, text, want in cases:
        ans, dt = m.ask(q + text)
        j = parse_json(ans)
        got = sorted(x.get("code") for x in j) if isinstance(j, list) and all(isinstance(x, dict) for x in j) else None
        structure_ok = got == sorted(want) and all(
            str(x.get("shelf")) == x["code"].split("-")[0] and str(x.get("level")) == x["code"].split("-")[1] for x in j)
        record("request", name, bool(structure_ok), ans, json.dumps(want, ensure_ascii=False), dt)

    # ---- 4. choose the next action ----------------------------------------------
    print("\n4. 根据当前情况选下一步动作（纯文字）")
    tools = ('你是帮用户在快递货架照片里找包裹的助手。可用动作只有这些：\n'
             'scan：分块扫描整张照片找取件码\n'
             'zoom_reread：把候选位置放大后重新读一遍\n'
             'mark：确认无误，在照片上圈出并告诉用户\n'
             'ask_retake：照片看不清，请用户重拍\n'
             'ask_other_shelf：这张照片不是目标所在的货架，请用户去正确的货架拍\n'
             '根据下面的情况选一个动作。只输出 JSON：{"action": "动作名"}\n\n情况：')
    cases = [
        ("拍对了货架，还没开始找", "用户要找 8-2-5726。照片的货架标牌显示这是 8 号货架第 2 层。还没有扫描过。", "scan"),
        ("拍错了货架", "用户要找 6-1-0498。照片的货架标牌显示这是 8 号货架第 2 层。", "ask_other_shelf"),
        ("标牌读不出来", "用户要找 8-2-5726。照片整体模糊，货架标牌和包裹标签上的字都读不出来。", "ask_retake"),
        ("两次读数不一致", "用户要找 8-2-5726。扫描时在某个位置读到 5726，但复核时同一位置读到的是 5728。", "zoom_reread"),
        ("两次读数一致", "用户要找 8-2-5726。扫描时在某个位置读到 5726，复核时同一位置读到的仍然是 5726。", "mark"),
        ("层号不对", "用户要找 8-4-3310。照片的货架标牌显示这是 8 号货架第 2 层和第 3 层。", "ask_other_shelf"),
    ]
    for name, text, want in cases:
        ans, dt = m.ask(tools + text, max_tokens=60)
        j = parse_json(ans)
        record("action", name, isinstance(j, dict) and j.get("action") == want, ans, want, dt)

    # ---- 5. can it point at a parcel directly? ----------------------------------
    print("\n5. 直接定位（参考项，不计入通过率）")
    q = ('照片里有很多贴着四位数字标签的快递。找到数字是 5726 的那个标签，'
         '给出它在图中的位置，用 0 到 1000 的相对坐标，只输出 JSON：{"box": [x1, y1, x2, y2]}')
    ans, dt = m.ask(q, photo)
    print(f"  answer ({dt:.1f}s): {' '.join(ans.split())[:260]}")
    print("  参考：5726 的标签实际大约在 [375, 520, 420, 560]（按 OCR 分块结果估计，非精确标注）")

    # ---- summary ----------------------------------------------------------------
    print("\n汇总")
    for g, label in [("sign", "读货架标牌"), ("quality", "判断照片能不能用"), ("request", "理解取件请求"), ("action", "选下一步动作")]:
        rs = [ok for gg, ok in results if gg == g]
        print(f"  {label}: {sum(rs)}/{len(rs)}")


if __name__ == "__main__":
    main()
