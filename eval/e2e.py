"""Drive the agent service the way the app will: say what to find, send photos, print the run.

    python e2e.py <server> "<what to find>" <photo> [<photo> ...]
"""
import json
import os
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
server, text, shots = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3:]
LABEL = {"think": "想    ", "act": "做    ", "see": "看到  ", "turn": "改主意", "ask": "问你  ", "done": "结果  ", "error": "错误  "}


def call(path, data=None, ctype="application/json"):
    if isinstance(data, dict):
        data = json.dumps(data).encode()
    headers = {"X-Key": os.environ.get("PARCEL_KEY", "")}      # set PARCEL_KEY when the service requires a key
    if data is not None:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(server + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


task = call("/api/tasks", {"text": text})
print(f"你: {text}")
print(f"它读到 {len(task['parcels'])} 个取件码（来源: {task['read_by']}）: " +
      ", ".join(f"{p['code']} → {p['shelf']} 号货架第 {p['level']} 层" for p in task["parcels"]))
print("计划: " + "; ".join(f"去 {g['shelf']} 号货架第 {g['level']} 层找 {'、'.join(g['tails'])}" for g in task["plan"]))

for shot in shots:
    force = shot.endswith("!")
    path = shot.rstrip("!")
    print(f"\n--- 拍照: {path}" + ("（要求仍在这张里找）" if force else ""))
    with open(path, "rb") as f:
        pid = call(f"/api/tasks/{task['id']}/photos" + ("?force=1" if force else ""), f.read(), "image/jpeg")["photo_id"]
    after, scanning = -1, 0
    while True:
        r = call(f"/api/photos/{pid}/events?after={after}")
        for e in r["events"]:
            after = e["i"]
            if e["kind"] == "progress":
                scanning = e["index"]
                continue
            print(f"  {e['t']:5.1f}s  {LABEL.get(e['kind'], e['kind'])}  {e['text']}")
        if not r["running"]:
            break
        time.sleep(0.4)
    s = r["summary"] or {}
    print(f"  => 调用本地模型 {s.get('calls')} 次，改主意 {s.get('turns')} 次，求助 {s.get('asks')} 次，用时 {s.get('seconds')} 秒；"
          f"圈出的照片 /api/photos/{pid}/annotated.jpg")
    print("  进度: " + ", ".join(f"{p['tail']}={p['status']}" for p in r["task"]["parcels"]))
    if r["task"]["done"]:
        print("  全部找齐")
