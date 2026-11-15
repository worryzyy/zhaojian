"""HTTP service between the app (uni-app x / H5) and the local model.

    python server.py [--port 8765] [--llm http://127.0.0.1:18500] [--key SECRET] [--host 0.0.0.0]

Routes (JSON unless noted; send the key as header X-Key or ?key= when --key is set):

    GET  /api/health
    POST /api/tasks                    {"text": "..."}        -> task (parcels + plan)
    GET  /api/tasks/<id>                                      -> task
    GET  /api/history                                         -> {"items": [...]}  parcels found so far, newest first
    POST /api/tasks/<id>/codes         {"text": "..."}        -> task with the new codes added
    POST /api/tasks/<id>/remove        {"code": "8-2-5726"}   -> task without that parcel
    POST /api/voice                    audio recording (multipart "file" or raw bytes)  -> {"text": "..."}
    POST /api/tasks/image[?task=<id>]  image of the pickup message (same body forms as photos)
                                                              -> task; codes are read from the picture
    POST /api/tasks/<id>/photos        image bytes | multipart "file" | {"image_base64": "..."}  [?force=1]
                                                              -> {"photo_id": ...}; the agent starts working
    GET  /api/photos/<id>/events?after=N                      -> {"events": [...], "running": bool}
    POST /api/photos/<id>/force                               -> re-run the same photo, ignoring a shelf mismatch
    GET  /api/photos/<id>/original.jpg   (image)
    GET  /api/photos/<id>/annotated.jpg  (image)
    GET  /                             the H5 client in ../web, if present

Only the Python standard library and Pillow are needed.
"""
import argparse
import base64
import io
import json
import os
import re
import threading
import time
import traceback
import urllib.parse
import uuid
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

import agent

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
WEB = os.path.join(os.path.dirname(HERE), "web")
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".jpg": "image/jpeg", ".png": "image/png", ".svg": "image/svg+xml", ".json": "application/json"}

tasks, photos = {}, {}
history = []                       # parcels found, newest first: the pickup record
model_lock = threading.Lock()      # the model server handles one request at a time
state_lock = threading.Lock()
cfg = {}


def save_state():
    """Write tasks and finished photo runs to disk, so a restart of this service does not lose the person's list."""
    with state_lock:
        snapshot = {"tasks": tasks, "photos": {k: v for k, v in photos.items() if not v["running"]}, "history": history}
        tmp = os.path.join(DATA, "state.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(DATA, "state.json"))


def load_state():
    path = os.path.join(DATA, "state.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            snapshot = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"state.json unreadable, starting empty: {e}", flush=True)
        return
    tasks.update(snapshot.get("tasks", {}))
    history.extend(snapshot.get("history", []))
    photos.update(snapshot.get("photos", {}))
    print(f"restored {len(tasks)} task(s), {len(photos)} photo run(s)", flush=True)


_asr = {"model": None}
asr_lock = threading.Lock()


def transcribe(audio):
    """Speech to text with a local Whisper model (CPU, so it does not compete with the vision model for the GPU).

    The model is loaded on first use. The prompt nudges it to write pickup codes as digits with hyphens;
    whatever it writes, agent.spoken_to_written() later normalises spoken forms such as 八杠二杠五七二六.
    """
    from faster_whisper import WhisperModel, decode_audio      # ImportError here means "not installed"
    path = os.path.join(DATA, "voice_" + uuid.uuid4().hex[:8])
    with open(path, "wb") as f:
        f.write(audio)
    try:
        samples = decode_audio(path, sampling_rate=16000)
        peak = float(abs(samples).max()) if samples.size else 0.0
        seconds = samples.size / 16000
        if peak < 0.004:
            # nothing but the noise floor: say so rather than pretend to have misheard
            print(f"{time.strftime('%H:%M:%S')} voice: {seconds:.1f}s, peak {peak:.4f} -> silent", flush=True)
            return "", peak
        # phone mics can record very quietly; bring the loudest point up to 90% before recognising
        samples = samples * min(0.9 / peak, 40.0)
        with asr_lock:
            if _asr["model"] is None:
                _asr["model"] = WhisperModel(cfg["asr"], device="cpu", compute_type="int8", download_root=cfg["asr_dir"])
            # No example codes in the prompt: on a recording of room noise Whisper repeats its prompt,
            # and example codes would then turn into parcels the person never asked for.
            # The voice-activity filter drops stretches with no speech before recognition.
            segments, _info = _asr["model"].transcribe(
                samples, language="zh", beam_size=5, vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt="以下是普通话，用简体中文书写，数字用阿拉伯数字。快递取件码写成“货架号-层号-四位数字”的格式，中间用短横线连接。")
            kept = [s for s in segments if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0]
            text = "".join(s.text for s in kept).strip()
        print(f"{time.strftime('%H:%M:%S')} voice: {seconds:.1f}s, peak {peak:.4f}, "
              f"{len(kept)} speech segment(s) -> {text!r}", flush=True)
        return text, peak
    finally:
        os.replace(path, os.path.join(DATA, "last_voice.bin"))     # kept for troubleshooting, overwritten each time


def public_task(t):
    left = [p for p in t["parcels"] if p["status"] != "found"]
    return {"id": t["id"], "parcels": t["parcels"], "plan": agent.make_plan(t["parcels"]),
            "photos": t["photos"], "done": bool(t["parcels"]) and not left, "left": len(left)}


def work(photo_id, force):
    ph = photos[photo_id]
    task = tasks[ph["task"]]

    def emit(kind, text, **extra):
        ph["events"].append({"i": len(ph["events"]), "kind": kind, "text": text, "t": round(time.time() - ph["t0"], 2), **extra})

    try:
        with model_lock:
            ph["t0"] = time.time()
            brain = agent.Brain(cfg["llm"], cfg["model"])
            image = Image.open(ph["original"])
            already = {p["code"] for p in task["parcels"] if p["status"] == "found"}
            ph["summary"] = agent.run_photo(brain, task, image, emit, ph["annotated"], force=force)
            ph["summary"]["seconds"] = round(time.time() - ph["t0"], 1)
            boxes = {m["tail"]: m["box"] for m in ph["summary"].get("marks", [])}
            for p in task["parcels"]:
                if p["status"] == "found" and p["code"] not in already:
                    history.insert(0, {"code": p["code"], "tail": p["tail"], "shelf": p["shelf"], "level": p["level"],
                                       "at": time.strftime("%Y-%m-%d %H:%M"), "photo": photo_id, "task": task["id"],
                                       "box": boxes.get(p["tail"])})
    except Exception as e:                       # the app must hear about a failure, not wait forever
        traceback.print_exc()
        emit("error", f"出错了：{type(e).__name__}: {e}")
    finally:
        ph["running"] = False
        save_state()


def start_photo(task_id, image_bytes, force=False):
    image = agent.normalise(Image.open(io.BytesIO(image_bytes)))
    pid = uuid.uuid4().hex[:10]
    original, annotated = os.path.join(DATA, pid + ".jpg"), os.path.join(DATA, pid + "_found.jpg")
    image.save(original, quality=92)
    photos[pid] = {"id": pid, "task": task_id, "original": original, "annotated": annotated, "events": [],
                   "running": True, "summary": None, "t0": time.time(), "size": list(image.size)}
    tasks[task_id]["photos"].append(pid)
    threading.Thread(target=work, args=(pid, force), daemon=True).start()
    return {"photo_id": pid, "size": list(image.size)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("%s %s" % (time.strftime("%H:%M:%S"), fmt % args), flush=True)

    # ---- plumbing
    def send(self, code, body, ctype="application/json; charset=utf-8", cache=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def fail(self, code, message):
        self.send(code, {"error": message})

    def body(self):
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def json_body(self):
        try:
            return json.loads(self.body() or b"{}")
        except json.JSONDecodeError:
            return {}

    def image_body(self):
        ctype, raw = self.headers.get("Content-Type", ""), self.body()
        if ctype.startswith("multipart/"):
            msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw)
            for part in msg.iter_parts():
                if part.get_filename() or part.get_content_maintype() == "image":
                    return part.get_payload(decode=True)
            return None
        if ctype.startswith("application/json"):
            b64 = (json.loads(raw or b"{}").get("image_base64") or "").split(",")[-1]
            return base64.b64decode(b64) if b64 else None
        return raw or None

    def allowed(self, query):
        key = cfg["key"]
        return not key or self.headers.get("X-Key") == key or query.get("key", [""])[0] == key

    def do_OPTIONS(self):
        self.send(204, b"")

    # ---- routes
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path, query = url.path, urllib.parse.parse_qs(url.query)
        if not path.startswith("/api/"):
            return self.static(path)
        if path == "/api/history":
            if not self.allowed(query):
                return self.fail(401, "需要访问口令")
            return self.send(200, {"items": history[:200]})
        if path == "/api/health":
            return self.send(200, {"ok": True, "model": cfg["model"], "key_required": bool(cfg["key"])})
        if not self.allowed(query):
            return self.fail(401, "需要访问口令")
        m = re.fullmatch(r"/api/tasks/(\w+)", path)
        if m:
            t = tasks.get(m.group(1))
            return self.send(200, public_task(t)) if t else self.fail(404, "没有这个任务")
        m = re.fullmatch(r"/api/photos/(\w+)/events", path)
        if m:
            ph = photos.get(m.group(1))
            if not ph:
                return self.fail(404, "没有这张照片")
            after = int(query.get("after", ["-1"])[0])
            return self.send(200, {"events": [e for e in ph["events"] if e["i"] > after], "running": ph["running"],
                                   "summary": ph["summary"], "task": public_task(tasks[ph["task"]]) if not ph["running"] else None})
        m = re.fullmatch(r"/api/photos/(\w+)/(original|annotated)\.jpg", path)
        if m:
            ph = photos.get(m.group(1))
            f = ph and ph[m.group(2)]
            if not f or not os.path.exists(f):
                return self.fail(404, "图片还没有生成")
            with open(f, "rb") as fh:
                return self.send(200, fh.read(), "image/jpeg")
        self.fail(404, "没有这个接口")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path, query = url.path, urllib.parse.parse_qs(url.query)
        if not self.allowed(query):
            return self.fail(401, "需要访问口令")
        try:
            self.post_route(path, query)
        finally:
            if path.startswith("/api/tasks"):
                save_state()

    def post_route(self, path, query):
        try:
            if path == "/api/tasks":
                text = str(self.json_body().get("text") or "")
                with model_lock:
                    parcels, source = agent.parse_request(agent.Brain(cfg["llm"], cfg["model"]), text)
                tid = uuid.uuid4().hex[:10]
                tasks[tid] = {"id": tid, "parcels": parcels, "photos": [], "text": text}
                return self.send(200, {**public_task(tasks[tid]), "read_by": source})
            if path == "/api/voice":
                # a short recording of the person saying what to find -> text, transcribed on this machine
                data = self.image_body()
                if not data:
                    return self.fail(400, "没有收到录音")
                try:
                    text, peak = transcribe(data)
                    return self.send(200, {"text": text, "level": round(peak, 4)})
                except ImportError:
                    return self.fail(501, "这台电脑上没有安装语音识别（faster-whisper）")
            if path == "/api/tasks/image":
                # a picture of the pickup message instead of typed text; ?task=<id> adds to an existing task
                data = self.image_body()
                if not data:
                    return self.fail(400, "没有收到图片")
                with model_lock:
                    found, read_text, source = agent.parse_request_image(
                        agent.Brain(cfg["llm"], cfg["model"]), Image.open(io.BytesIO(data)))
                tid = query.get("task", [""])[0]
                if tid in tasks:
                    t = tasks[tid]
                    have = {p["code"] for p in t["parcels"]}
                    added = [p for p in found if p["code"] not in have]
                    t["parcels"].extend(added)
                else:
                    tid = uuid.uuid4().hex[:10]
                    t = tasks[tid] = {"id": tid, "parcels": found, "photos": [], "text": read_text}
                    added = found
                return self.send(200, {**public_task(t), "added": [p["code"] for p in added],
                                       "read_by": source, "read_text": read_text})
            m = re.fullmatch(r"/api/tasks/(\w+)/remove", path)
            if m and m.group(1) in tasks:
                # the person crossed a parcel off the list (picked up already, or read wrong)
                t = tasks[m.group(1)]
                code = str(self.json_body().get("code") or "")
                t["parcels"] = [p for p in t["parcels"] if p["code"] != code]
                return self.send(200, public_task(t))
            m = re.fullmatch(r"/api/tasks/(\w+)/codes", path)
            if m and m.group(1) in tasks:
                t = tasks[m.group(1)]
                with model_lock:
                    more, source = agent.parse_request(agent.Brain(cfg["llm"], cfg["model"]), str(self.json_body().get("text") or ""))
                have = {p["code"] for p in t["parcels"]}
                added = [p for p in more if p["code"] not in have]
                t["parcels"].extend(added)
                return self.send(200, {**public_task(t), "added": [p["code"] for p in added], "read_by": source})
            m = re.fullmatch(r"/api/tasks/(\w+)/photos", path)
            if m and m.group(1) in tasks:
                data = self.image_body()
                if not data:
                    return self.fail(400, "没有收到照片")
                return self.send(200, start_photo(m.group(1), data, force=query.get("force", ["0"])[0] == "1"))
            m = re.fullmatch(r"/api/photos/(\w+)/force", path)
            if m and m.group(1) in photos:
                self.body()
                old = photos[m.group(1)]
                with open(old["original"], "rb") as fh:
                    return self.send(200, start_photo(old["task"], fh.read(), force=True))
            self.body()
            self.fail(404, "没有这个接口")
        except Exception as e:
            traceback.print_exc()
            self.fail(500, f"{type(e).__name__}: {e}")

    def static(self, path):
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        full = os.path.normpath(os.path.join(WEB, rel))
        if not full.startswith(WEB) or not os.path.isfile(full):
            return self.fail(404, "没有这个页面")
        with open(full, "rb") as fh:
            self.send(200, fh.read(), MIME.get(os.path.splitext(full)[1], "application/octet-stream"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--llm", default="http://127.0.0.1:18500", help="llama-server serving MiniCPM-V 4.7")
    ap.add_argument("--model", default="minicpm-v")
    ap.add_argument("--key", default=os.environ.get("PARCEL_KEY", ""), help="access key required from clients")
    ap.add_argument("--asr", default="small", help="Whisper model size for voice input (tiny, base, small, ...)")
    ap.add_argument("--asr-dir", default=os.path.join(HERE, "data", "whisper"), help="where the Whisper model is kept")
    a = ap.parse_args()
    cfg.update(llm=a.llm, model=a.model, key=a.key, asr=a.asr, asr_dir=a.asr_dir)
    os.makedirs(DATA, exist_ok=True)
    load_state()
    print(f"parcel finder on http://{a.host}:{a.port}  (model {a.model} at {a.llm}, key {'set' if a.key else 'not set'})")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
